# Protein Design Agent System
A local multi-agent system utilizing AutoGen and Ollama for protein design analysis, incorporating AlphaFold, CollabFold, and RFdiffusion workflows.

## Architecture
- **Coordinator Agent**: Oversees the design cycle.
- **Qwen Summarizer**: Processes original biological papers and extracts critical parameters.
- **DeepSeek Critic**: Compares the summaries directly against the original papers to eliminate hallucinations.

## Requirements
- Python 3.10+
- AutoGen v0.4+
- Local Ollama server
