import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from sklearn.model_selection import train_test_split
from app.preprocess import clean_text

#load dataset
df = pd.read_csv("data/scam_messages.csv")



#clean messages
df["message"]= df["message"].apply(clean_text)

df = df.dropna(subset=["label"])

X = df["message"]
Y = df["label"]


X_train,X_test,y_train,y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

print(f"Train size: {len(X_train)}")    
print(f"Test size: {len(X_test)}")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,f1_score
import joblib

#TF_IDF Vedctorization

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# MOdel 1: Logistic Regression
logreg = LogisticRegression(max_iter=1000, n_jobs=-1)
logreg.fit(X_train_vec,y_train)

log_preds = logreg.predict(X_test_vec)
print("\n logreg results")
print(classification_report(y_test,log_preds))
print("Weighted F1:", f1_score(y_test, log_preds, average="weighted"))

#model 2: linear SVM 

svm = LinearSVC()
svm.fit(X_train_vec, y_train)
svm_preds = svm.predict(X_test_vec)
print("\nLinear SVM Results")
print(classification_report(y_test, svm_preds))
print("Weighted F1:", f1_score(y_test, svm_preds, average="weighted"))

#Save Models
os.makedirs("models/artifacts",exist_ok=True)
joblib.dump(vectorizer,"models/artifacts/tfidf_vectorizer.pkl")
joblib.dump(svm,"models/artifacts/svm_classifier.pkl")

print("models saved")