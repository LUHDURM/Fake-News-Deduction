## Fake News Detection System

A machine learning web app built using **Python, Scikit-learn, and Streamlit** to classify news articles as **Real** or **Fake**.

---

## 📘 Project Overview

This project detects fake news by analyzing the text content of news articles.  
It uses **TF-IDF Vectorization** for feature extraction and **Logistic Regression** for classification.

---

## 🧩 Technologies Used

- Python 3.13
- Scikit-learn
- Pandas
- NumPy
- Seaborn / Matplotlib
- Streamlit
- Joblib

---

## Basic Information

- mkdir fake-news-detection
mkdir means “make directory” — it creates a new folder.

- cd fake-news-detection
cd means “change directory” — it moves you inside that folder you just created.

-  python3 -m venv venv
where,**python3** is the version of python and **-m** call a module like thing **venv** the first venv creates the virtual environment where the second one is the name that we assign for our virtual environment.So in short it creates our virtual environment that we need.

- source venv/bin/activate
This is the word which acivates our virtual environment.

- deactivate
This is the word which deactivates the virtual environment.

- pip install –upgrade pip
To make sure the pip upgrade is upto date.

---

## Dependencies 

- pip install numpy pandas scikit-learn matplotlib seaborn jupyterlab joblib

| Package      | Purpose                          | Needed for VS Code?                     |
| ------------ | -------------------------------- | --------------------------------------- |
| numpy        | Numerical operations             | Yes                                     |
| pandas       | Data loading & manipulation      | Yes                                     |
| scikit-learn | ML algorithms & metrics          | Yes                                     |
| matplotlib   | Plotting & visualization         | Yes                                     |
| seaborn      | Advanced plotting / prettier EDA | Optional but recommended                |
| jupyterlab   | Notebook environment             | Optional (VS Code can handle notebooks) |
| joblib       | Save/load trained ML models      | Yes                                     |


1️⃣ numpy

What it is: Fundamental package for numerical computing in Python.
Why you need it: Many libraries (like pandas and scikit-learn) rely on it for arrays, math operations, and matrix computations.
Do you need it? ✅ Yes, always useful for ML projects.

2️⃣ pandas

What it is: Library for data manipulation and analysis.
Why you need it: Lets you load CSV/Excel files, clean data, filter, group, and explore datasets easily.
Do you need it? ✅ Absolutely — your fake news dataset will probably be CSV, so pandas is essential.

3️⃣ scikit-learn (sklearn)

What it is: The main machine learning library for Python.
Why you need it: Provides classifiers (Logistic Regression, Random Forest, Naive Bayes), metrics (accuracy, F1, confusion matrix), and preprocessing tools (TF-IDF, scaling).
Do you need it? ✅ Yes — the core library for your ML models.

4️⃣ matplotlib

What it is: Library for creating plots and graphs.
Why you need it: Lets you visualize trends, distributions, and evaluation metrics (like confusion matrices).
Do you need it? ✅ Yes, for plotting charts and graphs for EDA and model evaluation.

5️⃣ seaborn

What it is: A higher-level plotting library built on matplotlib.
Why you need it: Makes beautiful statistical plots (like heatmaps, correlation plots) with less code.
Do you need it? ✅ Optional but highly recommended — makes EDA much easier and prettier.

6️⃣ jupyterlab

What it is: Interactive notebook environment for Python.
Why you need it: Lets you write code, run it, see outputs and plots all in one interface — perfect for EDA and step-by-step ML workflows.
Do you need it if using VS Code? ⚠️ Optional.
VS Code has built-in Jupyter support, so if you open .ipynb files in VS Code, you don’t strictly need to install jupyterlab.
But installing it doesn’t hurt — gives you the option to also run Jupyter notebooks in your browser if you want.

7️⃣ joblib

What it is: A library for saving and loading Python objects efficiently.
Why you need it: You can save your trained ML model to a file and load it later without retraining.
Do you need it? ✅ Yes — essential if you want to save your fake news classifier.


-  pip install nltk spacy xgboost imbalanced-learn flask

| Package          | Why use it                                   | Needed for baseline?    |
| ---------------- | -------------------------------------------- | ----------------------- |
| nltk             | Text preprocessing (stopwords, tokenization) | Optional, nice to have  |
| spacy            | Advanced NLP preprocessing                   | Optional                |
| xgboost          | Stronger ML model for classification         | Optional                |
| imbalanced-learn | Handle imbalanced dataset                    | Optional                |
| Streamlit        | Deploy model as web app / API                | Optional, only for demo |


1️⃣ nltk (Natural Language Toolkit)

What it is: A Python library for text processing and NLP.
Why use it:
Tokenization (splitting text into words or sentences)
Removing stopwords (common words like “the”, “is”)
Stemming and lemmatization (reducing words to base forms)
Frequency analysis (finding common words in fake vs real news)
Do you need it? ✅ Useful if you want deeper preprocessing beyond TF-IDF.

2️⃣ spacy

What it is: Another NLP library, faster and more modern than NLTK.
Why use it:
Lemmatization, part-of-speech tagging
Named entity recognition (detecting people, places, organizations)
Can be used for advanced preprocessing or transformer pipelines
Do you need it? ⚠️ Optional for your first baseline. Only use if you want cleaner text preprocessing.

3️⃣ xgboost

What it is: Gradient boosting library for machine learning.
Why use it:
Produces very accurate classification models
Handles large datasets efficiently
Can outperform Logistic Regression and Random Forest in some cases
Do you need it? Optional — only if you want to try a stronger classifier after your baseline models.

4️⃣ imbalanced-learn (imblearn)

What it is: Library for handling imbalanced datasets.
Why use it:
In fake news detection, datasets may have more “real” news than “fake”
Provides oversampling techniques (SMOTE) and undersampling
Helps your model learn better and improve recall for the minority class
Do you need it? ⚠️ Optional for now, but useful if you notice class imbalance.

5️⃣ Streamlit

What it is: Streamlit is a Python framework that makes it super easy to build interactive web apps directly from your data science or machine learning code — without needing HTML, CSS, or JavaScript.


- pip freeze > requirements.txt
List all installed Python packages in my current environment and save that list into a file called requirements.txt.

---

## Code Block by block explanation

- # Step 1: Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split                                                            
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

- # Step 2: Load and Combine Datasets

'''fake_df = pd.read_csv("Dataset/Fake.csv")
true_df = pd.read_csv("Dataset/train.csv")'''

- pd.read_csv() → This function from the pandas library is used to read data from a CSV (Comma-Separated Values) file and convert it into a DataFrame (a table-like structure in Python).

- fake_df → Stores the data from Fake.csv, which contains fake news articles.

- true_df → Stores the data from train.csv, which contains true or real news articles

- # Add labels: 1 = Fake, 0 = Real

'''fake_df["label"] = 1
true_df["label"] = 0'''

- fake_df["label"] = 1 → Adds a new column named ‘label’ to the fake news dataset and assigns the value 1 to all its rows.
→ This means all entries in fake_df are fake news.

- true_df["label"] = 0 → Adds the same column to the true news dataset but assigns the value 0.
→ This means all entries in true_df are real news.

- A new column named ‘label’ is added to both datasets — assigning 1 for fake news and 0 for real news — so the model can learn to classify them correctly.

- # Combine both

'''data = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)
print(" Data Loaded and Combined Successfully!")
print(data.head())'''

- pd.concat([fake_df, true_df], axis=0) → Combines the fake and true news DataFrames vertically (row-wise) to form one single dataset called data.
axis=0 means stacking one below the other.

- .reset_index(drop=True) → Resets the index numbers so they run continuously after merging, without keeping the old index values.

- print(data.head()) → Displays the first five rows of the combined dataset for a quick look at the data

- Both fake and real news datasets are merged into a single DataFrame using pd.concat(). The index is reset, and the first few records are displayed to verify successful combination.

- # Step 3: Basic Data Information

print("\nData Info:")
print(data.info())
print("\nMissing Values:")
print(data.isnull().sum())
print("\nLabel Counts:")
print(data['label'].value_counts())

- data.info() → Shows a quick summary of the dataset, including:
The number of rows and columns
Column names
Data types (e.g., object, int64)
Non-null (non-missing) entries

- data.isnull().sum() → Checks for missing or empty values in each column and counts how many are missing.

- data['label'].value_counts() → Counts how many records are Fake (1) and Real (0) in the dataset.

- This step displays the dataset’s structure, checks for missing values, and counts the number of fake and real news samples to understand the data before training.

- # Step 4: Data Cleaning , Fill missing values

'''data = data.fillna('')'''

- data.fillna('') → Replaces all missing or null values in the dataset with an empty string ('').
This ensures that there are no blank cells which could cause errors during text processing or model training.
The cleaned data is stored back into data.

- Missing values in the dataset are replaced with empty strings using fillna('') to keep the data clean and ready for analysis.

# Combine title and text if both exist

'''if 'title' in data.columns and 'text' in data.columns:
    data['content'] = data['title'] + " " + data['text']
elif 'text' in data.columns:
    data['content'] = data['text']
else:
    raise ValueError("Dataset must contain at least a 'text' column!")'''

- Checks if the dataset has both title and text columns:
If yes, it combines them into a new column called content, separating them with a space.
This helps the model use all available information from the article.
If only the text column exists, it uses text alone as content.
If neither text nor title exists, it raises an error, because at least some text is required for training.
The new column content is what the model will use as input for fake news detection.

- A new column content is created by combining title and text (if both exist) so that the model can use all textual information for training.

# Features (X) and Labels (y)

'''X = data['content']
y = data['label']'''

- X = data['content'] → Selects the text data (title + content) as the features for the model.
Features are the input data the model will learn from.

- y = data['label'] → Selects the label column (1 = Fake, 0 = Real) as the target/output for the model.

- The content column is used as input features (X), and the label column is used as target labels (y) for model training.

- # Step 5: Train-Test Split

'''X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nData Split Complete!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))'''

- train_test_split() → Splits the dataset into training and testing sets.
Training set (X_train, y_train) → Used to train the model.
Testing set (X_test, y_test) → Used to evaluate model performance.

- test_size=0.2 → 20% of the data is used for testing, and 80% for training.

- random_state=42 → Ensures the split is reproducible (same every time).

- stratify=y → Maintains the same proportion of fake and real news in both training and testing sets.

- The dataset is split into 80% training and 20% testing, keeping the proportion of fake and real news the same, so the model can be trained and evaluated effectively.

- # Step 6: Text Vectorization (TF-IDF)

'''vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("\n TF-IDF Vectorization Complete!")
print("Shape of X_train_tfidf:", X_train_tfidf.shape)'''

- TfidfVectorizer → Converts text into numerical features that a machine learning model can understand.
Each word gets a weight based on importance in the document relative to the entire dataset.

- stop_words='english' → Ignores common English words like “the”, “and”, “is” which are not useful for prediction.

- max_df=0.7 → Ignores words that appear in more than 70% of the documents (very common words).

- fit_transform(X_train) → Learns the vocabulary from the training data and converts it into a TF-IDF matrix.

- transform(X_test) → Converts the test data into the same TF-IDF space without re-learning the vocabulary.

- .shape → Shows the number of samples and features (words) in the training set.

- The text data is converted into numerical features using TF-IDF, which assigns importance scores to words. This prepares the data for machine learning.

- # Step 7: Train the Model

'''model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
print("\n Model Training Complete!")'''

- LogisticRegression(max_iter=1000) → Creates a logistic regression model, a type of machine learning algorithm commonly used for binary classification (Fake vs Real).
max_iter=1000 ensures the model has enough iterations to converge during training.

- model.fit(X_train_tfidf, y_train) → Trains the model using the TF-IDF features (X_train_tfidf) and labels (y_train).

- A logistic regression model is trained on the TF-IDF features and corresponding labels to learn how to classify news as fake or real.

- # Step 8: Predictions & Evaluation

'''y_pred = model.predict(X_test_tfidf)
print("\n🔍 Evaluation Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))'''

- model.predict(X_test_tfidf) → Uses the trained model to predict labels (Fake or Real) for the test dataset.

- accuracy_score(y_test, y_pred) → Calculates the overall accuracy of the model: how many predictions match the actual labels.

- classification_report(y_test, y_pred) → Provides detailed metrics:
Precision → How many predicted fakes were actually fake
Recall → How many actual fakes were correctly predicted
F1-score → Balance between precision and recall

- The trained model predicts labels for the test set, and its performance is evaluated using accuracy and a classification report with precision, recall, and F1-score.

- # Step 9: Confusion Matrix Visualization

'''cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real','Fake'], yticklabels=['Real','Fake'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()'''

- confusion_matrix(y_test, y_pred) → Creates a matrix that shows:
True Positives (TP) → Real news predicted as Real
True Negatives (TN) → Fake news predicted as Fake
False Positives (FP) → Real news predicted as Fake
False Negatives (FN) → Fake news predicted as Real

- sns.heatmap() → Visualizes the confusion matrix as a color-coded table for easier understanding.
annot=True → Shows the numbers inside the cells
fmt='d' → Formats numbers as integers
xticklabels & yticklabels → Label axes as Real/Fake

- plt.show() → Displays the plot.

- A confusion matrix is plotted to visualize how many news articles were correctly or incorrectly classified as fake or real, giving insight into the model’s performance.

- # Step 10: Save Model and Vectorizer

'''joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\n Model and Vectorizer Saved!")'''

- joblib.dump(model, "fake_news_model.pkl") → Saves the trained logistic regression model to a file called fake_news_model.pkl.

- joblib.dump(vectorizer, "vectorizer.pkl") → Saves the TF-IDF vectorizer to vectorizer.pkl.

- The trained model and TF-IDF vectorizer are saved using joblib so they can be loaded later for predictions without retraining.

- # Step 11: Test the Model on New Input

'''def predict_news(news_text):
    vectorized_text = vectorizer.transform([news_text])
    prediction = model.predict(vectorized_text)[0]
    return "Real News" if prediction == 0 else " Fake News"'''

- def predict_news(news_text): → Defines a function to predict whether a given news article is real or fake.

- vectorizer.transform([news_text]) → Converts the new input text into TF-IDF features, just like the training data.

- model.predict(vectorized_text)[0] → Uses the trained model to predict the label (0 = Real, 1 = Fake) for the input.

- return "Real News" if prediction == 0 else "Fake News" → Returns a human-readable result based on the predicted label.

- A function predict_news is created to test new articles. It converts the input text into TF-IDF features and uses the trained model to predict whether the news is real or fake.

---

