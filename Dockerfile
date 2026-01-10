FROM python:3.10-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Clone the repository
WORKDIR /code

# Copy the necessary files
COPY database/pipeline.pkl /code/database/pipeline.pkl
COPY pyproject.toml /code/pyproject.toml
COPY knowledge_database /code/knowledge_database
COPY api /code/api

# Install Python dependencies using uv
RUN uv pip install --system .

# Set up the secret environment variable for OpenAI API Key
RUN --mount=type=secret,id=OPENAI_API_KEY sh -c 'echo "export OPENAI_API_KEY=$(cat /run/secrets/OPENAI_API_KEY)" >> /etc/profile.d/openai.sh'

# Set the command to run the application
CMD ["/bin/bash", "-c", "source /etc/profile && uvicorn api.api:app --host 0.0.0.0 --port 8080"]
