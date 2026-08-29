FROM python:3.12-slim 
#1. python runtime as parent image

# These make Python logs appear immediately in Docker and keep generated
# bytecode out of the application source directory.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/model-cache

WORKDIR /app 
#2. define working directory inside container

# 3.copy dependency file (to leverage docker caching)
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt
#4. install production dependencies

#5. copy rest of application source code
COPY app ./app
COPY knowledge_base ./knowledge_base

RUN mkdir -p /app/data /app/model-cache

#6. document the port the container listens on at runtime
EXPOSE 8000

#7. define the command to run the application when the container starts
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
