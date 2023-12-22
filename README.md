# llama-pc
based on ggerganov/llama.cpp

## LLM model
- llama-2–7b-chat.Q2_K.gguf
source: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main
baidu disk: https://pan.baidu.com/s/1YvAYrDD6DfoxpwD2kT5n3w?pwd=1234
## Docker
```commandline 
docker build -t songgs/llm-cpu -f deploy/Dockerfile .
docker run -it -p 8000:8000 songgs/llm-cpu
```
## Note
Run model CPU costs a lot of resources.
## refer
- https://medium.com/@penkow/how-to-run-llama-2-locally-on-cpu-docker-image-731eae6398d1