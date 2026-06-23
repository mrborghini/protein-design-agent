import argparse
import os
import sys
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core.models import ModelInfo
from autogen_core import CancellationToken

parser = argparse.ArgumentParser()
parser.add_argument('--host', default='http://localhost:11434', help='Ollama host URL')
args, unknown_args = parser.parse_known_args()
user_prompt = " ".join(unknown_args)
if not user_prompt:
    user_prompt = "Summarize recent findings on RFdiffusion for binder design."

ollama_host = args.host

model_info = ModelInfo(
    vision=False,
    function_calling=True,
    json_output=True,
    family="unknown"
)

qwen_client = OllamaChatCompletionClient(
    model="qwen3.5:latest",
    host=ollama_host,
    model_info=model_info
)

gemma_client = OllamaChatCompletionClient(
    model="gemma4:latest",
    host=ollama_host,
    model_info=model_info
)


gpt_oss_client = OllamaChatCompletionClient(
    model="gpt-oss:latest",
    host=ollama_host,
    model_info=model_info
)


def web_search(query: str) -> str:
    """Search the web using SearXNG."""
    import urllib.request, urllib.parse, json
    try:
        url = f"http://localhost:8080/search?q={urllib.parse.quote(query)}&format=json"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            results = [f"{r.get('title', '')}: {r.get('content', '')}" for r in data.get('results', [])[:3]]
            return "\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search failed: {e}"

def read_file(filepath: str) -> str:
    """Read text from a local file."""
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def save_file(filepath: str, content: str) -> str:
    """Save text content to a local file."""
    try:
        with open(filepath, 'w') as f:
            f.write(content)
        return f"Successfully saved to {filepath}"
    except Exception as e:
        return f"Error saving file: {e}"

async def main():

    literature_agent = AssistantAgent(
        name="LiteratureAgent",
        model_client=qwen_client,
        tools=[web_search, read_file, save_file],
        system_message="You extract key facts about protein design. You can search the web and read files."
    )
    
    hypothesis_agent = AssistantAgent(
        name="HypothesisAgent",
        model_client=gemma_client,
        tools=[read_file, save_file],
        system_message="You generate actionable hypotheses for protein design. You can read and save files."
    )
    
    critic = AssistantAgent(
        name="Critic",
        model_client=gpt_oss_client,
        tools=[save_file],
        system_message="You critique hypotheses based on facts. Save your final conclusions to a file."
    )

    print(">>> Starting Protein Design Cycle with Native AutoGen v0.4 Ollama Client <<<")
    
    token = CancellationToken()
    try:
        reply = await literature_agent.on_messages(
            [TextMessage(content=user_prompt, source="user")],
            cancellation_token=token
        )
        print(f"\n[Literature Agent Summary]:\n{reply.chat_message.content}")
        
        hyp_reply = await hypothesis_agent.on_messages(
            [TextMessage(content=f"Based on this, propose a hypothesis:\n{reply.chat_message.content}", source="user")],
            cancellation_token=token
        )
        print(f"\n[Hypothesis Agent]:\n{hyp_reply.chat_message.content}")
        
        crit_reply = await critic.on_messages(
            [TextMessage(content=f"Critique this hypothesis:\n{hyp_reply.chat_message.content}", source="user")],
            cancellation_token=token
        )
        print(f"\n[Critic]:\n{crit_reply.chat_message.content}")
        
    except Exception as e:
        print(f"Error during execution: {e}")

if __name__ == "__main__":
    asyncio.run(main())
