import os
from autogen import ConversableAgent

# Configure Ollama endpoints (ensure OLLAMA_HOST is set to allow connections)
ollama_url = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434/v1")

config_qwen = {
    "config_list": [{"model": "qwen2.5:latest", "base_url": ollama_url, "api_key": "ollama"}]
}

config_deepseek = {
    "config_list": [{"model": "deepseek-r1:latest", "base_url": ollama_url, "api_key": "ollama"}]
}

# Initialize Agents
coordinator = ConversableAgent(
    name="Coordinator",
    system_message="You manage the workflow. Present the scientific paper to the Summarizer first. Once summarized, forward BOTH the original paper and the summary to the Critic.",
    llm_config=config_qwen,
    human_input_mode="NEVER"
)

summarizer = ConversableAgent(
    name="Summarizer",
    system_message="You extract biological data and model configurations from protein papers.",
    llm_config=config_qwen,
    human_input_mode="NEVER"
)

critic = ConversableAgent(
    name="Critic",
    system_message="You are a rigorous validator. Compare the summary against the original paper text to catch and correct hallucinations.",
    llm_config=config_deepseek,
    human_input_mode="NEVER"
)

def run_design_cycle(paper_text: str):
    print("--- Starting Guided Design Cycle ---")
    
    # 1. Summary Phase
    summary_prompt = f"Please summarize the following paper text: \n\n{paper_text}"
    summary_reply = summarizer.generate_reply(messages=[{"content": summary_prompt, "role": "user"}])
    print("\n[Summarizer Reply]:\n", summary_reply)
    
    # 2. Direct Criticism Phase (passing both paper and summary to DeepSeek to prevent hallucination)
    critic_prompt = f"Original Paper Text:\n{paper_text}\n\nProposed Summary:\n{summary_reply}\n\nPlease verify if this summary has any hallucinations or errors."
    critic_reply = critic.generate_reply(messages=[{"content": critic_prompt, "role": "user"}])
    print("\n[DeepSeek Critic Verification]:\n", critic_reply)

if __name__ == "__main__":
    sample_paper = "RFdiffusion uses a denoising diffusion probabilistic model to generate backbone structures."
    run_design_cycle(sample_paper)
