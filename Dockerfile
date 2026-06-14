# 1. Use the official lightweight Python image as the base
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the dependency list and install
COPY ./python/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy all source code from the current directory into the container
COPY . .

# 5. Declare the port the container listens on
EXPOSE 8000

# 6. Switch the working directory into the python folder so uvicorn can locate main.py
WORKDIR /app/python/src

# 7. Startup command: bind uvicorn to 0.0.0.0 to allow external connections
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
