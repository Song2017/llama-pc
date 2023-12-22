# Use python as base image
FROM python:3.12-bullseye

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./llama_pc/server.py /app/server.py
COPY ./requirements.txt /app/requirements.txt
COPY ./huggingface/llama-2-7b-chat.Q2_K.gguf /app/llama-2-7b-chat.Q2_K.gguf

# Install the needed packages
RUN set -ex \
    && mkdir -p .pip \
    && pip3 --cache-dir=.pip install \
       -i https://mirrors.aliyun.com/pypi/simple/ -r /app/requirements.txt

EXPOSE 8000

# Run llama_cpu_server.py when the container launches
CMD ["python", "/app/server.py"]