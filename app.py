#!/usr/bin/env python3
"""
Gradio wrapper app to convert safetensors / Hugging Face diffusers into GGUF and/or ONNX,
with environment & version collection for reproducibility.

Usage:
  pip install -r requirements.txt
  python app.py
"""

import gradio as gr
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
import shlex
import sys
import platform
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi
from importlib import metadata

# -------------------------
# DEFAULT COMMAND TEMPLATES
DEFAULT_GGUF_CMD_LLM = (
    "python /path/to/llama.cpp/tools/convert.py --model {INPUT_PATH} --outfile {OUTPUT_PATH} --format gguf"
)
DEFAULT_ONNX_CMD_SDXL = (
    "python /path/to/convert_diffusers_to_onnx.py --repo {INPUT_DIR} --outdir {OUTPUT_DIR} --opset {OPSET}"
)
DEFAULT_GGUF_CMD_GENERIC = "echo 'Please edit GGUF command template in the UI to a real converter' && false"
DEFAULT_ONNX_CMD_GENERIC = "echo 'Please edit ONNX command template in the UI to a real converter' && false

# Packages to probe by default (you can change this list in code)
DEFAULT_PACKAGES = [
    "gradio",
    "huggingface_hub",
    "transformers",
    "diffusers",
    "onnx",
    "onnxruntime",
    "optimum",
    "bitsandbytes",
    "accelerate",
]

# -------------------------
# Utilities
def run_and_stream(cmd, cwd=None):
    """
    Run command (string) and yield stdout/stderr lines for Gradio streaming.
    Returns exit code at the end of the generator.
    """
    if isinstance(cmd, str):
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            shell=True,
            universal_newlines=True,
            bufsize=1,
            executable="/bin/bash",
        )
    else:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd, universal_newlines=True, bufsize=1)

    output_accum = []
    for line in iter(process.stdout.readline, ""):
        if line is None:
            break
        output_accum.append(line)
        yield "".join(output_accum)
    process.stdout.close()
    return_code = process.wait()
    yield "".join(output_accum) + f"\n[PROCESS EXIT CODE {return_code}]\n"
    return return_code

def zip_dir(dir_path, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dir_path):
            for f in files:
                absf = os.path.join(root, f)
                arcname = os.path.relpath(absf, dir_path)
                zf.write(absf, arcname)
    return zip_path

def get_package_version(pkg_name):
    try:
        return metadata.version(pkg_name)
    except metadata.PackageNotFoundError:
        return None
    except Exception:
        return None

def run_version_command(cmd, cwd=None, timeout=30):
    """
    Run a single version-check command and return (success, output).
    """
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, cwd=cwd, timeout=timeout, executable="/bin/bash")
        return True, out.decode("utf-8", errors="replace")
    except subprocess.CalledProcessError as e:
        return False, e.output.decode("utf-8", errors="replace") if e.output else str(e)
    except Exception as e:
        return False, str(e)

def collect_environment_info(outputs_dir, hf_model_id=None, extra_version_cmds=None, packages=None):
    """
    Collect environment and version information and write to environment.txt in outputs_dir.
    Returns a string with the summary.
    """
    lines = []
    lines.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Platform: {platform.platform()}")
    lines.append(f"Python: {sys.version.replace(os.linesep, ' ')}")
    try:
        lines.append(f"CPU count: {os.cpu_count()}")
    except Exception:
        pass

    # pip / git versions
    ok, out = run_version_command("python -V")
    if ok:
        lines.append(f"python -V: {out.strip()}")
    ok, out = run_version_command("pip --version")
    if ok:
        lines.append(f"pip --version: {out.strip()}")
    ok, out = run_version_command("git --version")
    if ok:
        lines.append(f"git --version: {out.strip()}")

    # Selected package versions
    pkgs = packages or DEFAULT_PACKAGES
    lines.append("\nPython package versions:")
    for p in pkgs:
        v = get_package_version(p)
        lines.append(f"  {p}: {v if v is not None else '<not installed>'}")

    # Extra version-check commands (one per line)
    lines.append("\nExtra version-check commands output:")
    if extra_version_cmds:
        for i, cmd in enumerate([c.strip() for c in extra_version_cmds.splitlines() if c.strip()]):
            lines.append(f"--- CMD #{i+1}: {cmd}")
            ok, out = run_version_command(cmd)
            if ok:
                lines.append(out.strip())
            else:
                lines.append(f"[FAILED] {out.strip()}")
    else:
        lines.append("  (none)")

    # HF model/sha info (if provided)
    if hf_model_id:
        try:
            api = HfApi()
            info = api.model_info(hf_model_id)
            # model_info contains 'sha' or 'sha' in revision property on some versions
            sha = getattr(info, "sha", None) or getattr(info, "sha256", None) or getattr(info, "pipeline_tag", None)
            lines.append(f"\nHugging Face model: {hf_model_id}")
            lines.append(f"  model_info (repr): {info}")
            if sha:
                lines.append(f"  reported sha/identifier: {sha}")
        except Exception as e:
            lines.append(f"\nFailed to retrieve HF model info for {hf_model_id}: {e}")

    # Write to file
    env_path = Path(outputs_dir) / "environment.txt"
    try:
        env_path.write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass

    return "\n".join(lines), str(env_path)

# -------------------------
# Main Gradio function
def convert_pipeline(
    input_mode,
    uploaded_file,
    hf_repo_id,
    model_type,
    want_gguf,
    want_onnx,
    gguf_template,
    onnx_template,
    opset,
    quant_option,
    collect_env,
    extra_version_cmds,
):
    start_ts = time.strftime("%Y%m%d-%H%M%S")
    work_root = Path(tempfile.mkdtemp(prefix=f"convert-{start_ts}-"))
    outputs_dir = work_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    log_accum = []

    def append_log(s):
        log_accum.append(s)
        return "".join(log_accum)

    yield "Preparing workspace...\n"

    input_path = None
    input_dir = None
    model_id_used = None
    try:
        # Collect environment info early if requested
        if collect_env:
            yield "Collecting environment and version information...\n"
            env_summary, env_file = collect_environment_info(outputs_dir, hf_model_id=(hf_repo_id.strip() if hf_repo_id else None), extra_version_cmds=extra_version_cmds)
            yield env_summary + "\n"
            yield f"Environment info written to: {env_file}\n"

        # Prepare input
        if input_mode == "upload":
            if uploaded_file is None:
                yield "No file uploaded. Aborting.\n"
                return
            # Normalize gradio uploaded file shape
            saved = None
            # gr.File commonly gives a dict with 'name'/'tmp_path' or a tuple; handle common cases
            try:
                if isinstance(uploaded_file, (list, tuple)) and len(uploaded_file) >= 1:
                    # usually (filename, filepath)
                    possible = uploaded_file[-1]
                    if os.path.exists(possible):
                        saved = Path(possible)
                elif hasattr(uploaded_file, "name") and os.path.exists(uploaded_file.name):
                    saved = Path(uploaded_file.name)
                elif hasattr(uploaded_file, "tmp_path") and os.path.exists(uploaded_file.tmp_path):
                    saved = Path(uploaded_file.tmp_path)
            except Exception:
                pass

            if saved is None:
                # fallback: try to write bytes to a file
                try:
                    tmpf = work_root / "uploaded.bin"
                    with open(tmpf, "wb") as f:
                        f.write(uploaded_file.read())
                    saved = tmpf
                except Exception as e:
                    yield f"Failed to save uploaded file: {e}\n"
                    return

            if saved.suffix in [".zip"]:
                extract_dir = work_root / "uploaded_extracted"
                extract_dir.mkdir()
                try:
                    shutil.unpack_archive(str(saved), str(extract_dir))
                    input_dir = str(extract_dir)
                    input_path = str(saved)
                    yield f"Uploaded archive extracted to {input_dir}\n"
                except Exception as e:
                    yield f"Failed to extract archive: {e}\n"
                    return
            else:
                input_path = str(saved)
                input_dir = None
                yield f"Uploaded file saved at {input_path}\n"

        else:  # huggingface_repo
            if not hf_repo_id or hf_repo_id.strip() == "":
                yield "No Hugging Face repo id provided. Aborting.\n"
                return
            model_id_used = hf_repo_id.strip()
            yield f"Downloading {model_id_used} from Hugging Face (this may take a while)...\n"
            try:
                repo_dir = snapshot_download(repo_id=model_id_used, local_dir=work_root / "hf_repo", repo_type="model")
                input_dir = str(repo_dir)
                input_path = None
                yield f"Downloaded repo to {input_dir}\n"
                # After download, if collect_env was not requested earlier, add model info now
                if not collect_env:
                    try:
                        api = HfApi()
                        info = api.model_info(model_id_used)
                        yield f"Hugging Face model info: {info}\n"
                    except Exception as e:
                        yield f"Failed to fetch model info: {e}\n"
            except Exception as e:
                yield f"Failed to download {model_id_used}: {e}\n"
                return

        # Now run conversions based on selections
        final_zip = outputs_dir / f"converted_{start_ts}.zip"
        did_any = False

        if want_gguf:
            did_any = True
            yield append_log("\n=== Starting GGUF conversion ===\n")
            gguf_out = outputs_dir / "model.gguf"
            tpl = gguf_template.strip() or DEFAULT_GGUF_CMD_GENERIC
            cmd = tpl.format(
                INPUT_PATH=input_path or input_dir or "",
                INPUT_DIR=input_dir or input_path or "",
                OUTPUT_DIR=str(outputs_dir),
                OUTPUT_PATH=str(gguf_out),
                OPSET=opset,
                QUANT=quant_option or "",
                MODEL_ID=model_id_used or "",
            )
            yield append_log(f"Running: {cmd}\n")
            for out in run_and_stream(cmd):
                yield out

        if want_onnx:
            did_any = True
            yield append_log("\n=== Starting ONNX conversion ===\n")
            onnx_outdir = outputs_dir / "onnx"
            onnx_outdir.mkdir(exist_ok=True)
            tpl = onnx_template.strip() or DEFAULT_ONNX_CMD_GENERIC
            cmd = tpl.format(
                INPUT_PATH=input_path or input_dir or "",
                INPUT_DIR=input_dir or input_path or "",
                OUTPUT_DIR=str(onnx_outdir),
                OUTPUT_PATH=str(onnx_outdir),
                OPSET=opset,
                QUANT=quant_option or "",
                MODEL_ID=model_id_used or "",
            )
            yield append_log(f"Running: {cmd}\n")
            for out in run_and_stream(cmd):
                yield out

        if not did_any:
            yield "No conversion target selected. Aborting.\n"
            return

        # Ensure environment.txt exists (collect if not already done)
        if not (outputs_dir / "environment.txt").exists():
            env_summary, env_file = collect_environment_info(outputs_dir, hf_model_id=(model_id_used or hf_repo_id or None), extra_version_cmds=extra_version_cmds)
            yield f"\nWrote environment summary to {env_file}\n"

        # Zip outputs for download
        yield "\nPackaging outputs...\n"
        zip_dir(str(outputs_dir), str(final_zip))
        yield f"Packaging complete. Download zip created: {final_zip}\n"
        yield f"[DONE]\n"
        return "".join(log_accum), str(final_zip)

    finally:
        # Keep workspace for inspection by default. Uncomment to auto-clean:
        # shutil.rmtree(work_root, ignore_errors=True)
        pass

# -------------------------
# Gradio UI
with gr.Blocks(title="Safetensors / Diffusers -> GGUF/ONNX converter (with version capture)") as demo:
    gr.Markdown(
        """
        # Model converter wrapper (with environment/version capture)
        This app wraps external converter tools. It can record environment/package/command versions for reproducibility.
        Edit the command templates to match your installed converters.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_mode = gr.Radio(choices=["upload", "huggingface_repo"], value="upload", label="Input mode")
            uploaded = gr.File(label="Upload safetensors or archive (zip)")
            hf_repo = gr.Textbox(label="Hugging Face repo id (e.g. runwayml/stable-diffusion-xl)", placeholder="owner/model-name")
            model_type = gr.Dropdown(choices=["LLM", "SDXL"], value="LLM", label="Model type / target domain")
            want_gguf = gr.Checkbox(value=True, label="Produce GGUF")
            want_onnx = gr.Checkbox(value=False, label="Produce ONNX")
            opset = gr.Number(value=16, label="ONNX opset (if producing ONNX)")
            quant_option = gr.Textbox(value="", label="Quantization option string (passed to templates)", placeholder="e.g. int8 or --quantize")
            collect_env = gr.Checkbox(value=True, label="Collect environment & version info")
            extra_version_cmds = gr.Textbox(label="Extra version-check commands (one per line)", placeholder="e.g. python /path/to/convert.py --version\n/path/to/gguf_quant --version", lines=4)
            convert_button = gr.Button("Convert")

        with gr.Column(scale=1):
            gr.Markdown("### Command templates (edit to match your converters)")
            gguf_template = gr.Textbox(label="GGUF command template", value=DEFAULT_GGUF_CMD_LLM, lines=3)
            onnx_template = gr.Textbox(label="ONNX command template", value=DEFAULT_ONNX_CMD_SDXL, lines=3)
            gr.Markdown(
                """
                Placeholders: {INPUT_PATH}, {INPUT_DIR}, {OUTPUT_DIR}, {OUTPUT_PATH}, {OPSET}, {QUANT}, {MODEL_ID}.
                Example GGUF (llama.cpp convert script) : 
                python /path/to/llama.cpp/tools/convert.py --safetensors {INPUT_PATH} --outfile {OUTPUT_PATH} --gguf
                """
            )

    log_output = gr.Textbox(label="Conversion log (streamed)", interactive=False, lines=20)
    download_file = gr.File(label="Download: zipped outputs")

    convert_button.click(
        convert_pipeline,
        inputs=[input_mode, uploaded, hf_repo, model_type, want_gguf, want_onnx, gguf_template, onnx_template, opset, quant_option, collect_env, extra_version_cmds],
        outputs=[log_output, download_file],
    )

if __name__ == "__main__":
    demo.launch()
