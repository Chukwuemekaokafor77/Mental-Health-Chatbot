import random
from model import ChatbotModel
from preprocessing import load_data, preprocess_text

class Chatbot:
    def __init__(self):
        self.data = load_data('../data/processed_mental_health_dataset.json')
        self.model = self.load_model('../models/chatbot_model.pkl')

    def load_model(self, file_path):
        return ChatbotModel.load(file_path)

    def get_response(self, user_input):
        # Preprocess the user input
        processed_input = preprocess_text(user_input)
        
        # Predict the intent
        intent = self.model.predict(processed_input)
        
        # Find the matching intent in our data
        for item in self.data['intents']:
            if item['tag'] == intent:
                # Return a random response from the intent
                return random.choice(item['responses'])
        
        return "I'm sorry, I didn't understand that. Can you please rephrase?"

if __name__ == "__main__":
    chatbot = Chatbot()
    print("Chatbot: Hello! How can I help you today? (Type 'quit' to exit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        response = chatbot.get_response(user_input)
        print("Chatbot:", response)