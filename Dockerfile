FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
# Download Presidio NLP model
RUN python -m spacy download en_core_web_lg 
COPY src/ ./src/
EXPOSE 8501
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]