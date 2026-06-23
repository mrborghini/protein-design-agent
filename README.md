# Protein Design Agent System

A local multi-agent system utilizing AutoGen and Ollama for protein design analysis, incorporating AlphaFold, CollabFold, and RFdiffusion workflows.

## Architecture

- **Coordinator Agent**: Oversees the design cycle.
- **Qwen Summarizer & Teacher**: Processes original biological papers and extracts critical parameters.
- **DeepSeek Critic**: Compares the summaries directly against the original papers to eliminate hallucinations.
- **Hypothesis Agent**: Proposes testable mutations and experiments.
- **Literature Agent**: Retrieves paper chunks via RAG.

## Prerequisites

- Python 3.10+
- Local [Ollama](https://ollama.com/) installation.
- Pull the required models:
  ```bash
  ollama pull qwen2.5
  ollama pull deepseek-coder
  ```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/mrborghini/protein-design-agent.git
   cd protein-design-agent
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Ensure your local Ollama server is running. If you are accessing Ollama over Tailscale or another network interface, set:
```bash
export OLLAMA_HOST=0.0.0.0
```

## Running the Agent

To start the agentic loop:
```bash
python agent.py
```
