from flask import Flask, render_template, request, jsonify
from src.model import ChatbotModel
from src.preprocessing import preprocess_text, load_data
import os
import random
from fuzzywuzzy import fuzz
import logging

logging.basicConfig(filename='chatbot.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

app = Flask(__name__, 
            static_folder=os.path.abspath('static'),
            template_folder=os.path.abspath('templates'))

class ChatbotSession:
    def __init__(self, model, intents_data):
        self.model = model
        self.intents_data = intents_data
        self.context = None

    def fuzzy_pattern_match(self, user_input, patterns, threshold=80):
        for pattern in patterns:
            if fuzz.partial_ratio(user_input.lower(), pattern.lower()) >= threshold:
                return True
        return False

    def get_response(self, user_input):
        processed_input = preprocess_text(user_input)
        intent, confidence = self.model.predict(processed_input)
        
        print(f"Predicted intent: {intent} (confidence: {confidence:.2f})")
        
        if confidence > 0.3:
            for intent_dict in self.intents_data['intents']:
                if intent_dict['tag'] == intent:
                    self.context = intent
                    return random.choice(intent_dict['responses'])
        
        for intent_dict in self.intents_data['intents']:
            if self.fuzzy_pattern_match(user_input, intent_dict['patterns']):
                self.context = intent_dict['tag']
                return random.choice(intent_dict['responses'])
        
        general_responses = [
            "Mental health is a crucial part of overall well-being. How can I provide more specific information about mental health?",
            "There are many aspects to mental health. What particular area would you like to know more about?",
            "Mental health affects how we think, feel, and act. Is there a specific aspect you're interested in?",
            "Taking care of our mental health is just as important as taking care of our physical health. What would you like to know more about?"
        ]
        return random.choice(general_responses)

# Load the data and train the model
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'data', 'processed_mental_health_dataset.json')
intents_data = load_data(data_path)

# Prepare data for training
X = []
y = []
for intent in intents_data['intents']:
    for pattern in intent['patterns']:
        X.append(pattern)
        y.append(intent['tag'])

chatbot_model = ChatbotModel()
chatbot_model.train(X, y)

chatbot_session = ChatbotSession(chatbot_model, intents_data)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json['user_input']
    logging.info(f"User input: {user_input}")
    response = chatbot_session.get_response(user_input)
    logging.info(f"Bot response: {response}")
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)