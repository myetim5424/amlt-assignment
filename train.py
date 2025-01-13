import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("processed_dataset.csv")

X = df['processed_review']
y = df['sentiment']  # Target variable (sentiment)

# check how many fields are nan
print(df.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# check the shape of the train and test datasets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# check the first 5 records of the train dataset
print(X_train.head())

from sklearn.feature_extraction.text import CountVectorizer

# Initialize CountVectorizer
vectorizer = CountVectorizer(max_features=5000, stop_words='english')

# Initialize TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')

# Fit and transform the training data
X_train_bow = vectorizer.fit_transform(X_train.values.astype('U'))
X_test_bow = vectorizer.transform(X_test)

# Fit and transform the training data
X_train_tfidf = tfidf.fit_transform(X_train.values.astype('U'))
X_test_tfidf = tfidf.transform(X_test)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Train the model
model_bow = LogisticRegression(max_iter=1000)
model_bow.fit(X_train_bow, y_train)

# Make predictions
y_pred_bow = model_bow.predict(X_test_bow)

print(y_pred_bow)

# Evaluate the model
print("Accuracy of bow:", accuracy_score(y_test, y_pred_bow))
print(classification_report(y_test, y_pred_bow))

# Train the model
model_tfidf = LogisticRegression(max_iter=1000)
model_tfidf.fit(X_train_tfidf, y_train)

# Make predictions
y_pred_tfidf = model_tfidf.predict(X_test_tfidf)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred_tfidf))
print(classification_report(y_test, y_pred_tfidf))

from sklearn.linear_model import LogisticRegression,SGDClassifier

# training the linear svm model
svm=SGDClassifier(loss='hinge',max_iter=500,random_state=42)

svm.fit(X_train_bow, y_train)

y_pred_svm_bow = svm.predict(X_test_bow)

# training the linear svm model
svm=SGDClassifier(loss='hinge',max_iter=500,random_state=42)

svm.fit(X_train_tfidf, y_train)

# Make predictions
y_pred_svm_tfidf = svm.predict(X_test_tfidf)

# Evaluate the model
print("Accuracy of svm bow:", accuracy_score(y_test, y_pred_svm_bow))
print(classification_report(y_test, y_pred_svm_bow))

# Evaluate the model
print("Accuracy of svm tfidf:", accuracy_score(y_test, y_pred_svm_tfidf))
print(classification_report(y_test, y_pred_svm_tfidf))

# word cloud for positive review words plt.figure(figsize=(20, 20))
# Separate positive and negative reviews
positive_reviews = ' '.join(df[df['sentiment'] == 'positive']['review'])
negative_reviews = ' '.join(df[df['sentiment'] == 'negative']['review'])

from wordcloud import WordCloud

# Generate WordCloud for positive reviews
positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews)

# Generate WordCloud for negative reviews
negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_reviews)

import matplotlib.pyplot as plt

# # Plot the WordClouds
plt.figure(figsize=(10, 6))

# Positive WordCloud
plt.subplot(1, 2, 1)
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Positive Reviews WordCloud')

# Negative WordCloud
plt.subplot(1, 2, 2)
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Negative Reviews WordCloud')

plt.tight_layout()
plt.show()