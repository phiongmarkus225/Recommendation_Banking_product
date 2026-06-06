FROM python:3.10-slim

WORKDIR /workspace

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all directories required for execution
COPY api/ ./api
COPY app/ ./app
COPY src/ ./src
COPY data/ ./data
COPY models/ ./models

EXPOSE 8000

# Command to run FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
