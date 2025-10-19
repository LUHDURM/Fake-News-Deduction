# 📘 Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split                                                            
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# 📘 Step 2: Load and Combine Datasets
fake_df = pd.read_csv("Dataset/Fake.csv")
true_df = pd.read_csv("Dataset/train.csv")

# Add labels: 1 = Fake, 0 = Real
fake_df["label"] = 1
true_df["label"] = 0

# Combine both
data = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)
print(" Data Loaded and Combined Successfully!")
print(data.head())

# 📘 Step 3: Basic Data Information
print("\nData Info:")
print(data.info())
print("\nMissing Values:")
print(data.isnull().sum())
print("\nLabel Counts:")
print(data['label'].value_counts())

# 📘 Step 4: Data Cleaning
# Fill missing values
data = data.fillna('')

# Combine title and text if both exist
if 'title' in data.columns and 'text' in data.columns:
    data['content'] = data['title'] + " " + data['text']
elif 'text' in data.columns:
    data['content'] = data['text']
else:
    raise ValueError("Dataset must contain at least a 'text' column!")

# Features (X) and Labels (y)
X = data['content']
y = data['label']

# 📘 Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nData Split Complete!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 📘 Step 6: Text Vectorization (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("\n TF-IDF Vectorization Complete!")
print("Shape of X_train_tfidf:", X_train_tfidf.shape)

# 📘 Step 7: Train the Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
print("\n Model Training Complete!")

# 📘 Step 8: Predictions & Evaluation
y_pred = model.predict(X_test_tfidf)

print("\n🔍 Evaluation Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 📘 Step 9: Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real','Fake'], yticklabels=['Real','Fake'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 📘 Step 10: Save Model and Vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\n Model and Vectorizer Saved!")

# 📘 Step 11: Test the Model on New Input
def predict_news(news_text):
    vectorized_text = vectorizer.transform([news_text])
    prediction = model.predict(vectorized_text)[0]
    return "Real News" if prediction == 0 else " Fake News"

