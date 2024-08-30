from model import ChatbotModel
from preprocessing import preprocess_text, load_data
import os
import random
from fuzzywuzzy import fuzz

def load_model(file_path):
    return ChatbotModel.load(file_path)

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
        
        # Use predicted intent if confidence is high enough
        if confidence > 0.3:
            for intent_dict in self.intents_data['intents']:
                if intent_dict['tag'] == intent:
                    self.context = intent
                    return random.choice(intent_dict['responses'])
        
        # If confidence is low, try fuzzy matching with patterns
        for intent_dict in self.intents_data['intents']:
            if self.fuzzy_pattern_match(user_input, intent_dict['patterns']):
                self.context = intent_dict['tag']
                return random.choice(intent_dict['responses'])
        
        # If no match found, provide a general response about mental health
        general_responses = [
            "Mental health is a crucial part of overall well-being. It includes our emotional, psychological, and social well-being. How can I provide more specific information about mental health?",
            "There are many aspects to mental health, including emotional well-being, psychological state, and social connections. What particular area would you like to know more about?",
            "Mental health affects how we think, feel, and act. It also helps determine how we handle stress, relate to others, and make choices. Is there a specific aspect of mental health you're interested in?",
            "Taking care of our mental health is just as important as taking care of our physical health. This can involve seeking professional help, maintaining good relationships, and practicing self-care. What would you like to know more about?"
        ]
        return random.choice(general_responses)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'models', 'chatbot_model.pkl')
    data_path = os.path.join(current_dir, '..', 'data', 'processed_mental_health_dataset.json')
    
    chatbot_model = load_model(model_path)
    intents_data = load_data(data_path)
    
    chatbot_session = ChatbotSession(chatbot_model, intents_data)
    
    print("Chatbot is ready! Type 'quit' to exit.")
    print("This chatbot is designed to provide support and information about mental health concerns.")
    print("Remember, it's not a substitute for professional help.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        response = chatbot_session.get_response(user_input)
        print("Bot:", response)

print("Goodbye!")