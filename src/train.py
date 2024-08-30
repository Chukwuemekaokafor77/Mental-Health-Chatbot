from preprocessing import load_data, preprocess_text
from model import ChatbotModel
import os
import random

def augment_data(X, y):
    augmented_X = []
    augmented_y = []
    for x, intent in zip(X, y):
        augmented_X.append(x)
        augmented_y.append(intent)
        
        # Add slight variations
        words = x.split()
        if len(words) > 3:
            augmented_X.append(' '.join(random.sample(words, len(words))))
            augmented_y.append(intent)
        
        # Add a synonym (you might want to use a proper synonym library)
        synonyms = {
            'sad': 'unhappy',
            'happy': 'joyful',
            'angry': 'furious',
            'anxious': 'worried'
        }
        for word, synonym in synonyms.items():
            if word in x:
                augmented_X.append(x.replace(word, synonym))
                augmented_y.append(intent)
    
    return augmented_X, augmented_y

def train_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'processed_mental_health_dataset.json')
    data = load_data(data_path)
    
    X = [pattern for intent in data['intents'] for pattern in intent['patterns']]
    y = [intent['tag'] for intent in data['intents'] for _ in intent['patterns']]
    
    X, y = augment_data(X, y)
    
    model = ChatbotModel()
    model.train(X, y)
    return model

def save_model(model, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    model.save(file_path)

if __name__ == "__main__":
    trained_model = train_model()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'models', 'chatbot_model.pkl')
    save_model(trained_model, model_path)
    
    print("Model trained successfully and saved to 'models/chatbot_model.pkl'")