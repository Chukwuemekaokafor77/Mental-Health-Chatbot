import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords and add domain-specific stop words
    stop_words = set(stopwords.words('english'))
    stop_words.update(['feel', 'feeling', 'felt', 'im', 'ive', 'really', 'much', 'lot'])
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def preprocess_data(data):
    processed_data = {'intents': []}
    
    for intent in data['intents']:
        processed_intent = {
            'tag': intent['tag'],
            'patterns': [preprocess_text(pattern) for pattern in intent['patterns']],
        }
        if 'responses' in intent:
            processed_intent['responses'] = intent['responses']
        else:
            print(f"Warning: Intent '{intent['tag']}' is missing 'responses'. Using a default response.")
            processed_intent['responses'] = ["I understand you're talking about {0}. Can you tell me more about that?".format(intent['tag'])]
        
        processed_data['intents'].append(processed_intent)
    
    return processed_data

if __name__ == "__main__":
    data = load_data('data/mental_health_dataset.json')
    processed_data = preprocess_data(data)
    
    with open('data/processed_mental_health_dataset.json', 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    print("Data preprocessed successfully and saved to 'processed_mental_health_dataset.json'")