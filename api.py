import flask
from flask import request, jsonify
from Inference import Inference
from confLoader import ConfLoader

conf = ConfLoader()
app = flask.Flask(__name__)
app.config["DEBUG"] = conf.debug

try:
    prompt = open("prompt.txt", "r").read()
except FileNotFoundError:
    print("prompt.txt not found")
    prompt = "You are a helpful assistant that answer questions"

ai_model = Inference(load_in_4bit=conf.load_in_4bit, load_in_8bit=conf.load_in_8bit, huggingface_token=conf.token, model_name=conf.model, device=conf.device, torch_dtype=conf.torch_dtype,prompt=prompt)

@app.route('/generate', methods=['GET'])
def generate():
    if 'Token' not in request.headers:
        return jsonify({'error': 'No token in the request'}), 401

    token = request.headers.get('Token')
    
    if token != conf.api_token:
        return jsonify({'error': 'Invalid token'}), 401
 
    prompt = request.headers.get("request")
    print(prompt) #TODO: remove this line of shit bu
    
    if type(prompt) != str:
        return jsonify({'error': 'Invalid request'}), 400
    
    response, time, tokens = ai_model.generate(prompt,conf.max_new_tokens,conf.do_sample,conf.temperature,conf.top_k,conf.top_p)
    
    return jsonify()

if __name__ == "__main__":
    app.run(host=conf.host,port=conf.port)

