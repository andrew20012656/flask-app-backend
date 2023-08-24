import random
import torch
import json
import os
from . import nltk_utils
from . import neural_model

def load_model():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    training_data_path = os.path.join(script_directory, 'data.json')

    with open(training_data_path, 'r') as json_data:
        intents = json.load(json_data)
    
    FILE_PATH = os.path.join(script_directory, "data.pth")
    data = torch.load(FILE_PATH)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = neural_model.NeuralNet(input_size, hidden_size, output_size)
    model.load_state_dict(model_state)
    model.eval()

    return model, all_words, tags



def respond(model, prompt, tags, all_words):
    sentence = prompt

    script_directory = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_directory, 'data.json')

    with open(data_path, 'r') as json_data:
        intents = json.load(json_data)

    sentence = nltk_utils.tokenize(sentence)
    X = nltk_utils.bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)
    
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return f"{random.choice(intent['responses'])}"
    else:
        return f"I do not understand."
