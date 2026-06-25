# 1. Use the official lightweight Python image as the base
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install torch from the cu121 index FIRST: plain `pip install torch` (what
#    requirements.txt deliberately leaves unpinned, see its header comment)
#    resolves to a CUDA 13 build, which doesn't run on this host's CUDA 12.2
#    driver -- same constraint already documented in
#    python/scripts/requirements-feasibility.txt for the bare-metal venv.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Copy the dependency list and install everything else (torch/transformers
#    already satisfied by step 3, so this won't try to pull a CUDA 13 torch)
COPY ./python/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all source code from the current directory into the container
COPY . .

# 6. Declare the port the container listens on
EXPOSE 8000

# 7. Switch the working directory into the python folder so uvicorn can locate main.py
WORKDIR /app/python/src

# 8. Startup command: bind uvicorn to 0.0.0.0 to allow external connections
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
