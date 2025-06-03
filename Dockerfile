# Use lightweight Python base image
FROM python:3.10-slim

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Make your shell script executable
RUN chmod +x pc-value-estimator.sh

# Run the app using your shell script
CMD ["sh", "pc-value-estimator.sh"]
