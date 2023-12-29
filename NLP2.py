import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)



column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']

data = pd.read_csv("C:/Users/adham/Downloads/NLP_Data/Tweet_Sentiment.csv", encoding='latin-1', names = column_names)

data.info()

data.target

data.drop_duplicates()
data.drop(['ids', 'date', 'flag', 'user'], axis=1, inplace=True)

data.info()

X = data['text']
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())

model.fit(X_train, y_train)

labels = model.predict(X_test)

mat = confusion_matrix(y_test, labels)

sns.heatmap(mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 4', 'Class 0'], yticklabels=['Class 4', 'Class 0'])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()

print("Unique Predicted Labels:", np.unique(labels))
print("Unique True Labels:", np.unique(y_test))

def predict(s, model=model):
    pred = model.predict([s])
    if pred[0] == 0:
        return "Negative"
    else:
        return "Positive"

print(predict("This food was good and the staff was friendly"))

accuracy = accuracy_score(y_test, labels)

print("Accuracy:", accuracy)