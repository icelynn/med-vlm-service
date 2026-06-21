# Docker Setup Guide

Covers installing the official Docker Engine, building the image, and deploying and troubleshooting the FastAPI microservice container.
Back to overview: [Environment & Dependency Overview](./environment_and_dependencies_overview.md)

## 1. Setup

**Install the official stable Docker Engine**
```bash
# 0. Preventive cleanup: remove conflicting packages and stale repo config
sudo rm -f /etc/apt/sources.list.d/docker.list
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# 1. Create the keyring and download the official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# 2. Configure the official repo (lsb_release -cs resolves the system codename)
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 3. Install the Engine and core components
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin

# 4. Add the current user to the docker group (run docker without sudo)
sudo usermod -aG docker $USER
```
> **Important:** after this, run `exit` and reconnect via SSH for the group membership to take effect.

**Build and run the FastAPI microservice in the background**
```bash
# Get the source
git clone https://github.com/icelynn/med-vlm-service.git
cd med-vlm-service

# Build the image
docker build -t med-vlm-service .

# Run in the background. --env-file injects config at runtime,
# so updating keys or switching environments needs no rebuild.
docker run -d -p 8000:8000 --env-file .env --name medical_agent med-vlm-service
```

## 2. Troubleshooting

| Symptom | Root cause | Fix |
|---|---|---|
| `docker` requires sudo / permission error | Did not re-login after joining the docker group | `exit` and reconnect via SSH |
| `All connection attempts failed` | `httpx` is refused when opening an outbound TCP connection; commonly because `ENV=dev` points at Ollama but no Ollama service is listening on host port 11434 | Set `.env` to `ENV=local` (route via OpenRouter) and restart the container |
| `Ollama API error (HTTP 500)` | Schema mismatch: with `ENV=dev`, the code packages requests in Ollama's format, but the target API points back to OpenRouter, whose gateway cannot parse the non-OpenAI `image_url` nesting | Switch `.env` back to `ENV=local` to use the standard `_call_openrouter()` path |
| `exec: "curl": executable file not found in $PATH` | The lightweight `python:3.11-slim` base image ships without `curl`, so `docker exec ... curl` fails | Use the container's built-in Python as a network probe (see Verification) |

**Switch environment and restart the container**
```bash
nano .env                       # set ENV=local
docker rm -f medical_agent
docker run -d -p 8000:8000 --env-file .env --name medical_agent med-vlm-service
```

## 3. Verification

```bash
# 1. Confirm the container is Up
docker ps

# 2. Confirm the Docker version and permissions (a permission error means you did not re-login)
docker --version

# 3. Container-internal network / DNS health probe (replaces the missing curl)
docker exec -it medical_agent python -c "import socket; print(socket.gethostbyname('huggingface.co'))"
docker exec -it medical_agent python -c "import httpx; print('status code:', httpx.get('https://huggingface.co').status_code)"
# Probe 2 returning "status code: 200" confirms the container's internal network and DNS are healthy

# 4. End-to-end (E2E) test: log out of SSH and simulate an external client from your local machine
curl -X POST http://your-ec2-public-ip:8000/analyze \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "prompt=Please systematically evaluate this chest radiography for clinical assessment" \
  -F "image=@./data/test_images/normal_xray.jpg"
# Expected: standard JSON of the form {"report": "### Image Quality Assessment..."}
```
