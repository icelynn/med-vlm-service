FROM python:3.10-slim

WORKDIR /app

# Install torch from the cu121 index FIRST: plain `pip install torch` (left
# unpinned in requirements.txt, see its header comment) resolves to a CUDA 13
# build, which doesn't run on this host's CUDA 12.2 driver -- same constraint
# documented in python/scripts/requirements-feasibility.txt for the bare-metal venv.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# torch/transformers already satisfied above, so this won't pull a CUDA 13 torch
COPY ./python/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# uvicorn needs to run from python/src to locate main.py
WORKDIR /app/python/src

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
