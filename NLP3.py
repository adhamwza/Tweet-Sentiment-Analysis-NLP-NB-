import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
import re
import nltk

nltk.download('stopwords')
# Imports

# Due to the acquired data not having labeled columns, labels have been provided
column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']

# reading the data and their encoding and applying the column names
data = pd.read_csv("C:/Users/adham/Downloads/NLP_Data/Tweet_Sentiment.csv", encoding='latin-1',
                   names=column_names)

data.info()

# Drops all null value rows in the target and text columns
data.dropna(subset=['target'], inplace=True)
data.dropna(subset=['text'], inplace=True)


# Method that preforms all pre-processing required such as lower casing everything, removing non word elements and
# setting stop words

def preprocessing(txt):
    if pd.isna(txt):
        return ''  # Return an empty string for NaN values
    txt = txt.lower()
    txt = re.sub('[^a-zA-Z]', ' ', txt)
    txt = re.sub(r"\s+[a-zA-Z]\s+", ' ', txt)
    txt = re.sub(r'\s+', ' ', txt)
    words = txt.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    txt = ' '.join(words)
    return txt


# Adds a new column which has the text column but preprocessed
data['preprocessed_reviews'] = data['text'].apply(preprocessing)

# Sets the vectorizer which will weigh each word and performs max features in order
# not to use all memory
vectorization = TfidfVectorizer(stop_words='english', max_features=3000)
x = vectorization.fit_transform(data['preprocessed_reviews'])

BoW_reviews = pd.DataFrame(x.toarray(), columns=vectorization.get_feature_names_out())
print(BoW_reviews)

y = data['target']

# sets test size and training size and seed
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Defines model and trains it
model = MultinomialNB()
model.fit(x_train, y_train)

# Predicts on test data
labels = model.predict(x_test)

mat = confusion_matrix(y_test, labels)
sns.heatmap(mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()

print("Accuracy: ", accuracy_score(y_test, labels))


# Method that intakes strings and determines NB NLP on it
def predict(s):
    review = preprocessing(s)
    review_vectorized = vectorization.transform([review])
    return model.predict(review_vectorized)

s = " this food is good"
print("This review is predicted to be: ", "Positive" if predict(s) == 4 else "Negative")
