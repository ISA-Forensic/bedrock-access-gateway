FROM public.ecr.aws/docker/library/python:3.12-slim

WORKDIR /app

# Create data directory for SQLite databases
RUN mkdir -p /app/api/data

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy source including static UI
COPY . /app

ENV PORT=80

# Ensure data directory has proper permissions
RUN chmod 755 /app/api/data

# Volume for persistent data storage
VOLUME ["/app/api/data"]

CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT}"]
