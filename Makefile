.PHONY: install install-dev sync dev api run lint lint-fix check pre-commit pre-commit-install docker-build docker-run launch clean

# Install dependencies
install:
	uv sync --no-dev

# Install with dev dependencies
install-dev:
	uv sync --all-extras

# Sync dependencies (alias for install-dev)
sync:
	uv sync --all-extras

# Start local dev server
dev:
	uv run uvicorn api.api:app --reload --port 8000

# Start API server (production-like)
api:
	uv run uvicorn api.api:app --host 0.0.0.0 --port 8080

# Run the data extraction pipeline
run:
	uv run python run.py

# Linting and formatting check
lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy . --ignore-missing-imports

# Auto-fix linting issues
lint-fix:
	uv run ruff check --fix .
	uv run ruff format .

# Run all checks (lint + type check)
check: lint

# Run pre-commit hooks on all files
pre-commit:
	uv run pre-commit run --all-files

# Install pre-commit hooks
pre-commit-install:
	uv run pre-commit install

# Build Docker image
docker-build:
	echo ${OPENAI_API_KEY} > mysecret.txt
	docker build --secret id=OPENAI_API_KEY,src=mysecret.txt -t knowledge .
	rm -f mysecret.txt

# Run Docker container
docker-run:
	docker run -d --add-host host.docker.internal:host-gateway --name run_knowledge -p 8080:8080 knowledge

# Build and run Docker (legacy command)
launch: docker-build docker-run

# Stop and remove Docker container
docker-stop:
	docker stop run_knowledge || true
	docker rm run_knowledge || true

# Clean up
clean:
	rm -rf .venv
	rm -rf __pycache__
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
