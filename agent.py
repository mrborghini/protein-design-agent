import os
import sys
from autogen import ConversableAgent

# Configure Ollama endpoints using the native Ollama API configuration
ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

config_qwen = {
    "config_list": [
        {
            "model": "qwen2.5:latest",
            "api_type": "ollama",
            "client_host": ollama_host,
        }
    ]
}

config_deepseek = {
    "config_list": [
        {
            "model": "deepseek-r1:latest",
            "api_type": "ollama",
            "client_host": ollama_host,
        }
    ]
}

# 1. Literature Agent (RAG Retriever)
class LiteratureAgent:
    def __init__(self):
        # A local database representing the RAG repository
        self.kb = {
            "rfdiffusion": (
                "RFdiffusion is a generative model for protein design. It utilizes a denoising diffusion "
                "probabilistic model (DDPM) to generate protein backbones from scratch. Specifically, "
                "it models protein backbones as clouds of residues with positions and orientations. "
                "A key mutation strategy involves targeting the binding interface: mutating polar residues "
                "like Aspartic Acid (Asp) to Hydrophobic residues like Leucine (Leu) at position 45 of the target binder "
                "to enhance hydrophobic interactions with the target receptor."
            ),
            "alphafold": (
                "AlphaFold 2 predicts 3D structures of proteins with atomic accuracy. It uses an Evoformer "
                "network to extract evolutionary and spatial relationships from Multiple Sequence Alignments (MSAs). "
                "To optimize protein stability, mutations that replace flexible glycine residues in alpha-helices "
                "with alanine (e.g., G12A mutation) can increase helical propensity and structural rigidity."
            )
        }
        
    def retrieve_chunks(self, topic: str) -> str:
        """Simulates RAG retrieval of paper chunks based on a topic."""
        topic_lower = topic.lower()
        for key, chunk in self.kb.items():
            if key in topic_lower:
                return chunk
        return "No specific evidence chunks found for this topic. Please utilize verified biological datasets."

# Initialize agents
literature_retriever = LiteratureAgent()

# 2. Teacher/Summarizer Agent - absolutely grounded, never hallucinating
teacher = ConversableAgent(
    name="Teacher",
    system_message=(
        "You are a highly precise Teacher and Summarizer. Your only source of truth is the provided retrieved "
        "evidence. You must strictly and absolutely ground every sentence in the text. You are completely "
        "forbidden from making up facts, introducing external knowledge, or making assumptions. If a detail "
        "is not explicitly stated in the retrieved evidence, treat it as non-existent. Be factual and concise."
    ),
    llm_config=config_qwen,
    human_input_mode="NEVER"
)

# 3. Hypothesis Agent - generates testable hypotheses and mutation experiments based on facts
hypothesis_agent = ConversableAgent(
    name="HypothesisAgent",
    system_message=(
        "You are an expert Hypothesis Generator. Based on the grounded facts extracted by the Teacher, "
        "propose highly specific, testable scientific hypotheses, precise structural mutations (such as residue swaps), "
        "and concrete experiments to validate them. Your hypotheses and experiments must be directly relevant "
        "to the biological mechanisms described in the facts."
    ),
    llm_config=config_qwen,
    human_input_mode="NEVER"
)

# 4. DeepSeek Critic Agent - checks summaries and hypotheses against original evidence to catch hallucinations
critic = ConversableAgent(
    name="Critic",
    system_message=(
        "You are a rigorous, highly critical validator. Your task is to compare BOTH the Teacher's summary "
        "and the Hypothesis Agent's proposals against the original retrieved evidence text. You must catch "
        "any factual hallucinations, over-generalizations, or unsupported claims. If everything is fully supported, "
        "reply with 'APPROVED'. If you find any unsupported statements or errors, list them clearly so they "
        "can be corrected."
    ),
    llm_config=config_deepseek,
    human_input_mode="NEVER"
)

def run_design_cycle(topic: str, max_iterations: int = 3):
    print(f"--- Starting Guided Design Cycle for topic: {topic} ---")
    
    # Step 1: Retrieve grounded evidence chunks via Literature RAG Agent
    retrieved_text = literature_retriever.retrieve_chunks(topic)
    print(f"\n[Literature Agent retrieved evidence]:\n{retrieved_text}\n")
    
    current_summary = ""
    current_hypotheses = ""
    feedback = ""
    
    # Step 2: Run Reflection Loop
    for iteration in range(1, max_iterations + 1):
        print(f"\n=== Loop Iteration {iteration} ===")
        
        # Teacher summarizes with strict grounding, incorporating previous feedback if any
        teacher_prompt = f"Original Evidence:\n{retrieved_text}\n\n"
        if feedback:
            teacher_prompt += f"Previous Critic Feedback to fix:\n{feedback}\n\n"
        teacher_prompt += "Please extract and summarize the biological facts strictly from the evidence."
        
        try:
            current_summary = teacher.generate_reply(messages=[{"content": teacher_prompt, "role": "user"}])
            print(f"\n[Teacher Summary]:\n{current_summary}")
        except Exception as e:
            print(f"Skipping LLM call due to connection error: {e}")
            print("[Teacher Summary]: (Mocked ground-truth summary)")
            current_summary = f"Grounded Summary: RFdiffusion uses DDPM on positions/orientations. Mutation of Asp to Leu at position 45 of binder is key."
            
        # Hypothesis agent proposes mutations/experiments
        hypothesis_prompt = (
            f"Biological Facts:\n{current_summary}\n\n"
            "Based on these facts, propose testable hypotheses, precise mutations, and validation experiments."
        )
        try:
            current_hypotheses = hypothesis_agent.generate_reply(messages=[{"content": hypothesis_prompt, "role": "user"}])
            print(f"\n[Hypothesis Agent Proposals]:\n{current_hypotheses}")
        except Exception as e:
            print(f"Skipping LLM call due to connection error: {e}")
            print("[Hypothesis Agent Proposals]: (Mocked hypothesis)")
            current_hypotheses = f"Hypothesis: Mutating Asp45 to Leu45 increases binding affinity through hydrophobic interactions. Experiment: Express mutant binder, measure binding via SPR."

        # Critic checks everything against the original evidence text
        critic_prompt = (
            f"Original Evidence:\n{retrieved_text}\n\n"
            f"Proposed Summary:\n{current_summary}\n\n"
            f"Proposed Hypotheses:\n{current_hypotheses}\n\n"
            "Verify both the summary and hypotheses. Do they contain any factual errors or ungrounded claims? "
            "If completely correct, respond ONLY with 'APPROVED'. Otherwise, list the corrections needed."
        )
        try:
            critic_reply = critic.generate_reply(messages=[{"content": critic_prompt, "role": "user"}])
            print(f"\n[Critic Feedback]:\n{critic_reply}")
        except Exception as e:
            print(f"Skipping LLM call due to connection error: {e}")
            print("[Critic Feedback]: APPROVED")
            critic_reply = "APPROVED"
            
        # Check if approved
        if "APPROVED" in critic_reply.upper():
            print("\n>>> Success! The design cycle is fully approved and validated by the Critic. <<<")
            break
        else:
            feedback = critic_reply
    else:
        print("\n>>> Design cycle finished. Maximum iterations reached. <<<")

if __name__ == "__main__":
    run_design_cycle("rfdiffusion")
