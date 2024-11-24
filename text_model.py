import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import pickle

# Load the dataset
file_path = r'C:\Users\acer\Downloads\datasets\datasets\Text Dataset - Sheet1.csv'  # Use raw string
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())

df = df[['Column 4', 'Column 7']]

# Rename columns for simplicity
df.columns = ['review', 'label']

# Encode labels: 'Seller Issue' -> 0, 'Logistic Issue' -> 1
df['label'] = df['label'].map({'Seller Issue': 0, 'Logistic Issue': 1})

# Drop rows with missing values
df.dropna(inplace=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['label'], test_size=0.2, random_state=50)

# Text Vectorization using TF-IDF
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model Training using Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train_tfidf, y_train)

# Model Evaluation
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# Calculate and print additional metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Save the trained model and vectorizer in the current folder (same folder as the script)
with open('text_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Optionally, save the TF-IDF vectorizer in the same folder
with open('tfidf_vectorizer.pkl', 'wb') as tfidf_file:
    pickle.dump(tfidf, tfidf_file)

print("Model and TF-IDF vectorizer saved successfully!")
