import datetime

from flask import Flask, jsonify
from flask_restx import Api, Resource, fields
from llama_cpp import Llama

model = None
_model_path = "/Users/songgs/_git/llama-pc/huggingface/llama-2-7b-chat.Q2_K.gguf"
_system_message = "You are a helpful assistant"
_max_tokens = 500

# Create a Flask object
app = Flask("Llama server")
api = Api(app, version='1.0', title='miniGPT', description='Service')
ns = api.namespace('LLM GPT', description='LLM GPT')
todo = api.model('LLM', {
    'system_message': fields.String(description='system_message', default=_system_message),
    'user_message': fields.String(required=True, description='The task details', default="Please list 2 cat names")
})


@ns.route('/llm')
class Demo(Resource):
    def get(self):
        return "llm"

    @ns.expect(todo)
    def post(self):
        return generate_response(data={"user_message": api.payload.get("user_message")})


def generate_response(data: dict):
    global model
    start_time = datetime.datetime.now()
    try:
        system_message = data.get("system_message") or _system_message
        max_tokens = int(data['max_tokens']) if 'max_tokens' in data else _max_tokens

        # Check if the required fields are present in the JSON data
        if 'user_message' in data:
            user_message = data['user_message']

            # Prompt creation
            prompt = f"""<s>[INST] <<SYS>>
            {system_message}
            <</SYS>>
            {user_message} [/INST]"""
            # Create the model if it was not previously created
            if model is None:
                # Create the model
                model = Llama(model_path=_model_path)

            # Run the model
            output = model(prompt, max_tokens=max_tokens, echo=True)
            print(output, datetime.datetime.now())
            choice_text = output.get("choices", [{}])[0].get("text")
            choice_text.replace(prompt, "")
            return jsonify({
                "question": user_message,
                "timing": str(datetime.datetime.now() - start_time),
                "choice": choice_text.replace(prompt, "").lstrip(),
            })

        else:
            return jsonify({"error": "Missing required parameters"}), 400

    except Exception as e:
        return jsonify({"Error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
