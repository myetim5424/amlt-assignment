
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')


# Function to clean and preprocess the text
def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^A-Za-z\s]', '', text.lower())
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Return the cleaned text
    return " ".join(tokens)

def assign_sentiment_by_rating(rating):
    if rating >= 7:
        return 'positive'
    elif rating >= 4:
        return 'neutral'
    else:
        return 'negative'




df = pd.read_csv("drugsComTest_raw.csv")

# summary of the dataset
print(df.info())

print(df.columns)
print(len(df))

# get all the values you have in rating column
print(df['rating'].value_counts())

print(df.head())

# check if there are any missing values
print(df.isnull().sum())

# check how many records are there in the dataset
print(df.shape)

# remove the records with missing values
df = df.dropna()

# get all nan values in dataset
print(df.isnull().sum())

# preprocess the text
df['processed_review'] = df['review'].apply(preprocess_text)

# assign sentiment based on the rating
df['sentiment'] = df['rating'].apply(assign_sentiment_by_rating)

# check review column after preprocessing
print(df['review'])

# check processed_review column after preprocessing
print(df['processed_review'])

# # check sentiment column
# print(df['sentiment'])

# check how many records are there in the dataset
print(df.shape)

# remove the records with missing values
df = df.dropna()

# get all nan values in dataset
print("check if it has nan again")
print(df.isnull().sum())

# get all the values you have in sentiment column
print(df['sentiment'].value_counts())

df.to_csv("processed_dataset.csv", index=False)