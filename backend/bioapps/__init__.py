"""External structural-biology tools (bio-apps) wrapped as AutoGen FunctionTools.

Each module here wraps one tool that runs in its **own Python venv** via
``runner.run_in_env``. GPU tools acquire ``backend.gpu.gpu_exclusive`` so the
Ollama LLM is evicted from VRAM first (the 32 GB GPU can't hold the 120B model
*and* a structural-biology model at once). See SETUP_BIOAPPS.md for installation.

POC: real runs take minutes→hours and depend on externally-installed envs/weights;
failures surface as errors rather than being hidden.
"""
