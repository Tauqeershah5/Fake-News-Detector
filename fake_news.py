import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load datasets
fake = pd.read_csv("data/fake.csv")
real = pd.read_csv("data/true.csv")

# Add labels
fake['label'] = 0
real['label'] = 1

# Combine title and text
fake['text'] = fake['title'] + " " + fake['text']
real['text'] = real['title'] + " " + real['text']

# Merge and shuffle
data = pd.concat([fake[['text', 'label']], real[['text', 'label']]], axis=0)
data = data.sample(frac=1, random_state=42)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Sample test predictions
sample_news = [
    "NASA confirms Earth will be in total darkness for 15 days in November.",
    "U.S. military to accept transgender recruits starting Jan. 1.",
    "Donald Trump to be knighted by the Queen of England.",
    "COVID-19 vaccines are now being distributed globally.",
    "Aliens have signed a treaty with world leaders under Area 51."
]

print("News Credibility Results:\n")
for news in sample_news:
    prediction = model.predict(vectorizer.transform([news]))[0]
    result = "REAL" if prediction == 1 else "FAKE"
    print(f"- {news}\n  → {result}\n")