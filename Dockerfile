# Use official Python 3.11 image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl git build-essential && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Set Poetry config
ENV POETRY_VERSION=1.7.1
ENV PATH="/root/.local/bin:$PATH"
ENV POETRY_NO_INTERACTION=1
ENV PYTHONUNBUFFERED=1

# Copy and install dependencies
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root

# Copy app source code
COPY . .

# Expose port for FastAPI
EXPOSE 10000
# Run the app
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
