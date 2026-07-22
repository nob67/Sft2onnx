FROM python:3.11-slim

# Non-interactive
ENV PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install minimal system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       git \
       curl \
       build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps first for better caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the project
COPY . /app

# Create a non-root user and use it
RUN useradd -m appuser || true
RUN chown -R appuser:appuser /app
USER appuser

# Expose the Gradio default port
EXPOSE 7860
ENV PORT=7860

# Launch the Gradio `demo` object defined in app.py and bind to 0.0.0.0
# We import the module and call demo.launch(...) so the container starts the web UI
CMD ["bash", "-lc", "python -c \"import app; app.demo.launch(server_name='0.0.0.0', server_port=int(__import__('os').environ.get('PORT',7860)))\""]
