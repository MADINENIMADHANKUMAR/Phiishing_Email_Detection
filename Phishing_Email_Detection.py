import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("phishing_emails.csv")
class EmailFeatures(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, emails):
        features = []
        phishing_keywords = [
            "verify", "password", "bank",
            "urgent", "click", "login",
            "free", "winner", "account"
        ]
        for email in emails:
            email = str(email).lower()
            url_count = len(re.findall(r'http[s]?://', email))
            keyword_count = sum(
                keyword in email for keyword in phishing_keywords
            )
            email_length = len(email)
            features.append([
                url_count,
                keyword_count,
                email_length
            ])
        return pd.DataFrame(features)
tfidf = TfidfVectorizer(stop_words='english')
combined_features = FeatureUnion([
    ('tfidf', tfidf),
    ('email_features', EmailFeatures())
])
model = Pipeline([
    ('features', combined_features),
    ('classifier', LogisticRegression())
])
X_train, X_test, y_train, y_test = train_test_split(
    data['text'],
    data['label'],
    test_size=0.2,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Phishing', 'Safe'],
    yticklabels=['Phishing', 'Safe']
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
while True:
    print("\nEnter Email Text (or type quit):")
    user_email = input("> ")
    if user_email.lower() == "quit":
        break
    prediction = model.predict([user_email])[0]
    print("\nPrediction:", prediction.upper())