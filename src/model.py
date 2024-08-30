from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

class ChatbotModel:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

    def train(self, X, y):
        self.pipeline.fit(X, y)

    def predict(self, text):
        intent = self.pipeline.predict([text])[0]
        confidence = self.pipeline.predict_proba([text]).max()
        return intent, confidence

    def save(self, file_path):
        joblib.dump(self.pipeline, file_path)

    @classmethod
    def load(cls, file_path):
        model = cls()
        model.pipeline = joblib.load(file_path)
        return model