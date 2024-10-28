import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def load_data():
    data = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv", sep="\t", header=None, names=['label', 'text'])
    data['label'] = data['label'].apply(lambda x: 0 if x == "ham" else 1)
    return data

# TfidfVectorizer(Term Frequency-Inverse Document Frequency Vectorizer): Measures how frequently a term (word) appears in a document
def train_model(data):
    X = data['text']
    y = data['label']
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    joblib.dump(model, "spam_detector_model.pkl") 
    joblib.dump(vectorizer, "spam_detector_vectorizer.pkl")
    return accuracy, report

if __name__ == "__main__":
    data = load_data()
    accuracy, report = train_model(data)
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Classification report: {report}")
