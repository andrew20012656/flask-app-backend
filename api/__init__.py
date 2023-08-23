import os
import torch
import json
from flask import Flask, request
from api import chatbot, neural_model


def create_app():
    app = Flask(__name__)
    app.config['model'], app.config['tags'], app.config['all_words'] = setup()

    @app.route("/about")
    def about():
        return "about"
    
    @app.route("/home")
    def home():
        return "home"
    
    @app.route("/chat", methods=["POST"])
    def chat():
        user_prompt = request.get_json()['prompt']
        response = chatbot.respond(app.config['model'],
                       user_prompt,
                       app.config['tags'],
                       app.config['all_words'])
        return response
    
    return app


def setup():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_directory, 'data.json')

    with open(data_path, 'r') as json_data:
        intents = json.load(json_data)

    model_path = os.path.join(script_directory, 'data.pth')
    data = torch.load(model_path)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = neural_model.NeuralNet(
        input_size, hidden_size, output_size)
    model.load_state_dict(model_state)
    model.eval()

    return model, tags, all_words