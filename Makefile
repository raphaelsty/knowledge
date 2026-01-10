launch:
	echo ${OPENAI_API_KEY} > mysecret.txt
	docker build --secret id=OPENAI_API_KEY,src=mysecret.txt -t knowledge .
	docker run -d --add-host host.docker.internal:host-gateway --name run_knowledge -p 8080:8080 knowledge

local-dev-api:
	uvicorn api.api:app --reload

# Start local dev server using uv
dev:
	uv run uvicorn api.api:app --reload --port 8000

# Install dependencies with uv
install:
	uv pip install -r requirements.txt
