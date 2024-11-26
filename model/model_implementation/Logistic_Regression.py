# Import nødvendige biblioteker
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Last inn datasett (endre filbaner om nødvendig)
train_df = pd.read_csv("model/preprocessed_data/proccessed_train_en_feature_extracted.tsv", sep='\t')
dev_df = pd.read_csv("model/preprocessed_data/processed_dev_test_en.tsv", sep='\t')
test_df = pd.read_csv("model/preprocessed_data/processed_test_en_gold.tsv", sep='\t')

# Forbered data (hent tekst og etiketter)
X_train = train_df['sentence']
y_train = train_df['label']
X_dev = dev_df['sentence']
y_dev = dev_df['label']
X_test = test_df['sentence']
y_test = test_df['label']

# LabelEncoder for å konvertere tekstetiketter til tall
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_dev = label_encoder.transform(y_dev)
y_test = label_encoder.transform(y_test)

# TF-IDF feature extraction
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_dev_tfidf = vectorizer.transform(X_dev)
X_test_tfidf = vectorizer.transform(X_test)

# Initialiser og tren Logistisk Regresjon
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)

# Prediksjon på utviklingssettet
y_dev_pred = lr_model.predict(X_dev_tfidf)

# Evaluering på utviklingssettet
print("Development Set Evaluation:")
print(classification_report(y_dev, y_dev_pred, target_names=label_encoder.classes_))

# Prediksjon på testsettet
y_test_pred = lr_model.predict(X_test_tfidf)

# Evaluering på testsettet
print("Test Set Evaluation:")
print(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(cm)

# Visualiser forvirringsmatrisen
plt.figure(figsize=(8, 6))
plt.matshow(cm, cmap="Blues", alpha=0.8)
plt.colorbar()
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(x=j, y=i, s=cm[i, j], ha="center", va="center", color="black")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Logistic Regression")
plt.show()

# Accuracy
accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {accuracy:.2f}")
