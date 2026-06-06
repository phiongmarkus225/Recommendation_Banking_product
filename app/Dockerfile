FROM python:3.10-slim

WORKDIR /workspace

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend app and the customer dataset
COPY app/ ./app
COPY data/processed/clean_customer_dataNEW.csv ./data/processed/clean_customer_dataNEW.csv

EXPOSE 8501

ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Command to run Streamlit
CMD ["streamlit", "run", "app/main.py"]
