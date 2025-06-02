# Use an official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Set working directory
WORKDIR /app

# Copy all files into the container
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Ensure your launch script is executable
RUN chmod +x pc-value-estimator.sh

# Run Streamlit (which includes embedded Flask)
CMD ["sh", "pc-value-estimator.sh"]
