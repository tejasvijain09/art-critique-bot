# Use official Python slim image (lightweight)
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Copy app files from your local machine into container
COPY . .

# Upgrade pip first
RUN pip install --upgrade pip

# Install Python dependencies from requirements.txt
RUN pip install -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Command to run the app inside the container
ENTRYPOINT ["streamlit", "run", "streamlit_art.py", "--server.port=8501", "--server.address=0.0.0.0"]
