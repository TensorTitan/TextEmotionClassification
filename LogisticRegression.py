import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from keras.utils import to_categorical
from cleaned_tokenized import tokenized_cleaned

# Load the dataset
data = pd.read_csv(r"C:\Users\marty\Desktop\NLP project\archive\train.txt", sep=';')
test = pd.read_csv(r"C:\Users\marty\Desktop\NLP project\archive\test.txt", sep=';')

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(data['emotion'])
y_test = label_encoder.transform(test['emotion'])

# One-hot encode the emotion labels (if it's multi-class classification)
y_train = to_categorical(y_train, num_classes=6)  # Ensure you specify num_classes
y_test = to_categorical(y_test, num_classes=6)

# Tokenize and clean the reviews (custom tokenized_cleaned function assumed)
data['tokens'] = data['review'].apply(tokenized_cleaned)
test['tokens'] = test['review'].apply(tokenized_cleaned)
data['cleaned_review'] = data['tokens'].apply(lambda x: ' '.join(x))
test['cleaned_review'] = test['tokens'].apply(lambda x: ' '.join(x))

# Vectorize the text data using CountVectorizer (you can try TF-IDF or other methods)
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(data['cleaned_review']).toarray()
X_test = vectorizer.transform(test['cleaned_review']).toarray()

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000, multi_class='ovr', solver='lbfgs', random_state=42)
model.fit(X_train, np.argmax(y_train, axis=1))

# Predict on the test set
predictions = model.predict(X_test)

# Get classification metrics
predicted_classes = predictions
true_classes = np.argmax(y_test, axis=1)

print("Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=[str(i) for i in range(6)]))

print("Confusion Matrix:")
print(confusion_matrix(true_classes, predicted_classes))

model.save("LogisticRegression_emotion_detection.h5")