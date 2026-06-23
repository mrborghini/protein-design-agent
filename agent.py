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
args, _ = parser.parse_known_args()
ollama_host = args.host

model_info = ModelInfo(
    vision=False,
    function_calling=True,
    json_output=True,
    family="unknown"
)

qwen_client = OllamaChatCompletionClient(
    model="qwen3.5:9b",
    host=ollama_host,
    model_info=model_info
)

gemma_client = OllamaChatCompletionClient(
    model="gemma4:12b",
    host=ollama_host,
    model_info=model_info
)


gpt_oss_client = OllamaChatCompletionClient(
    model="gpt-oss:latest",
    host=ollama_host,
    model_info=model_info
)

async def main():
    literature_agent = AssistantAgent(
        name="LiteratureAgent",
        model_client=qwen_client,
        system_message="You extract key facts about protein design from literature."
    )
    
    hypothesis_agent = AssistantAgent(
        name="HypothesisAgent",
        model_client=gemma_client,
        system_message="You generate actionable hypotheses for protein design."
    )
    
    critic = AssistantAgent(
        name="Critic",
        model_client=gpt_oss_client,
        system_message="You critique hypotheses based on facts."
    )

    print(">>> Starting Protein Design Cycle with Native AutoGen v0.4 Ollama Client <<<")
    
    token = CancellationToken()
    try:
        reply = await literature_agent.on_messages(
            [TextMessage(content="Summarize recent findings on RFdiffusion for binder design.", source="user")],
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
