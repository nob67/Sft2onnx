# Sft2onnx

Safetensors and diffusers conversion to ONNX and GGUF for easier running on CPU-only devices. Made for Gradio servers.

This repository provides a Gradio-based wrapper (app.py) that helps you convert models provided as safetensors files or Hugging Face diffusers repositories into GGUF (for llama.cpp/llama.cpp-based runtimes) and/or ONNX (for CPU inference via ONNX Runtime). The app is a command-template wrapper — it shells out to the converter tools you have installed and records environment/version information for reproducibility.

## Contents

- `app.py` - Gradio app: upload or specify HF repo, choose outputs (GGUF / ONNX), edit command templates, run conversion, and download a zip of outputs. It captures environment info to `environment.txt` in the outputs.
- `requirements.txt` - Python packages commonly needed by the app (Gradio, huggingface_hub, diffusers, transformers, onnx, onnxruntime, optimum, bitsandbytes, accelerate).

## Quickstart

1. Clone the repo (or run inside this repository):

   git clone https://github.com/nob67/Sft2onnx.git
   cd Sft2onnx

2. Install Python dependencies (recommended in a venv):

   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

3. Edit converter command templates

   The app does not implement the conversion internals itself. Instead you must point it at converter scripts/tools that exist on your machine. Open `app.py` and edit the `DEFAULT_GGUF_CMD_LLM` and `DEFAULT_ONNX_CMD_SDXL` values, or edit the templates in the Gradio UI at runtime.

   Placeholders available for templates:
   - `{INPUT_PATH}` — path to the uploaded file (safetensors, etc.)
   - `{INPUT_DIR}` — path to a directory (e.g., downloaded HF repo or extracted archive)
   - `{OUTPUT_DIR}` — directory where outputs should be written
   - `{OUTPUT_PATH}` — concrete output file path (e.g. `outputs/model.gguf`)
   - `{OPSET}` — ONNX opset integer
   - `{QUANT}` — quantization option string (tool-specific)
   - `{MODEL_ID}` — HF repo id used (when providing a Hugging Face repo)

   Example template hints (replace with actual paths for your environment):
   - LLM safetensors -> GGUF (llama.cpp):
     `python /path/to/llama.cpp/tools/convert.py --safetensors {INPUT_PATH} --outfile {OUTPUT_PATH} --gguf`
   - Diffusers (SDXL) -> ONNX (example exporter script):
     `python /path/to/convert_diffusers_to_onnx.py --repo {INPUT_DIR} --outdir {OUTPUT_DIR} --opset {OPSET}`

4. Run the app:

   python app.py

   - The Gradio UI will open locally with a web interface. Upload a safetensors file or enter a Hugging Face repo id, choose conversion outputs, edit templates if needed, and press Convert.
   - Conversion logs stream live in the UI. After conversion the produced files and `environment.txt` are packaged into a zip for download.

## Environment & Version Capture

If enabled, the app writes an `environment.txt` into the outputs directory containing:
- OS and Python version information
- Selected Python package versions (configurable list in the code)
- Output from any ``Extra version-check commands`` you enter in the UI (test converter binaries, git SHAs, etc.)
- Hugging Face model info for repo inputs

This is intended to help with reproducibility and debugging conversion issues.

## Security & Resource Notes

- The app executes shell commands you provide in templates — only run this in trusted environments.
- Converting large models may require substantial RAM and disk space. CPU-only conversions can be slow and memory intensive.
- If converting private HF models, you must provide environment access to your HF token (the app currently uses `huggingface_hub.snapshot_download` which respects typical HF auth config such as `huggingface-cli login` or `HF_HOME` settings).

## Optional Improvements

If you want, I can:
- Pre-fill templates for specific converter tools you already have (tell me the tool names and paths).
- Add a UI field for a Hugging Face token for private models and automatically use it for downloads.
- Add automatic post-processing steps (e.g., ONNX quantization using onnxruntime tools or GGUF quant tools).
- Add an automated test or GitHub Actions workflow that runs a smoke test (small dummy model) to ensure the app starts correctly.

## Contributing

Patches welcome. Create a branch and a pull request. If you add templates or helper scripts for specific converters, please document the required tool versions and links to the converter project.

## License

Add a LICENSE file to declare a license. No license is included by default in this repository.
