import random
import torch
import json
import os
from . import nltk_utils

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
        return f"I do not understand..."
