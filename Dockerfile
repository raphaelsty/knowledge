FROM python:3.10

# Install dependencies
RUN apt-get update && apt-get install -y git git-lfs && git lfs install

# Clone the repository
WORKDIR /code

# Copy the necessary files (you may skip this if already in the repository)
COPY database/pipeline.pkl /code/pipeline.pkl
COPY requirements.txt /code/requirements.txt
COPY setup.py /code/setup.py
COPY knowledge_database /code/knowledge_database
COPY api /code/api

# Install Python dependencies
RUN pip install pip --upgrade
RUN pip install --no-cache-dir -r requirements.txt

# Set up the secret environment variable for OpenAI API Key
RUN --mount=type=secret,id=OPENAI_API_KEY sh -c 'echo "export OPENAI_API_KEY=$(cat /run/secrets/OPENAI_API_KEY)" >> /etc/profile.d/openai.sh'

# Set the command to run the application
CMD ["/bin/bash", "-c", "source /etc/profile && uvicorn api.api:app --host 0.0.0.0 --port 8080"]
