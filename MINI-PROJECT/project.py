# Step 1: Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 2: Load Dataset
df = pd.read_csv("IMDB.csv")  # Make sure this file is in your working directory
print("Sample Data:")
print(df.head())

# Step 3: Encode Sentiment Labels (positive -> 1, negative -> 0)
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Step 4: Split into Features and Labels
X = df['review']
y = df['label']

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Vectorize Text Data (Convert to numbers)
vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 7: Train the Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Step 8: Evaluate the Model
y_pred = model.predict(X_test_vec)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 9: Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show(block=False)  # Non-blocking plot

# Step 10: Predict Sentiment for a New Review
print("\n--- Predict Sentiment for a New Review ---")

# Hardcoded input for testing
new_review = input("Enter a review:")

new_review_vector = vectorizer.transform([new_review])
prediction = model.predict(new_review_vector)

print("Review:", new_review)
print("Predicted Sentiment:", "Positive 😊" if prediction[0] == 1 else "Negative 😞")
