# Use an official Python runtime as a parent image
FROM python:3.11-slim 
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"]
