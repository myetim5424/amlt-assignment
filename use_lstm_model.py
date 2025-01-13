from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("processed_dataset.csv")

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['review'])
sequences = tokenizer.texts_to_sequences(df['review'])

# Encode target labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(df['sentiment'])

# Load the model
loaded_model = load_model("lstm_patient_reviews.h5")

# Prepare new data
new_reviews = ["The medicine was very effective", "I had severe side effects", "It is very bad", "worst medicine ever"]
new_sequences = tokenizer.texts_to_sequences(new_reviews)
new_padded = pad_sequences(new_sequences, maxlen=100, padding='post', truncating='post')

# Predict
predictions = loaded_model.predict(new_padded)
predicted_classes = np.argmax(predictions, axis=1)
predicted_labels = label_encoder.inverse_transform(predicted_classes)
print(predicted_labels)
