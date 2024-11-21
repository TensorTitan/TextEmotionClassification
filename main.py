import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from ANN import create_ANN
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import to_categorical
from cleaned_tokenized import tokenized_cleaned
from sklearn.feature_extraction.text import CountVectorizer
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv(r"C:\Users\marty\Desktop\NLP project\archive\train.txt", sep=';')
test = pd.read_csv(r"C:\Users\marty\Desktop\NLP project\archive\test.txt", sep=';')

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(data['emotion'])
y_test = label_encoder.transform(test['emotion'])

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=6)
y_test = to_categorical(y_test, num_classes=6)

# Tokenization and cleaning
data['tokens'] = data['review'].apply(tokenized_cleaned)
test['tokens'] = test['review'].apply(tokenized_cleaned)
data['cleaned_review'] = data['tokens'].apply(lambda x: ' '.join(x))
test['cleaned_review'] = test['tokens'].apply(lambda x: ' '.join(x))

# Vectorization
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(data['cleaned_review']).toarray()
X_test = vectorizer.transform(test['cleaned_review']).toarray()

# Define EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

# Create and compile the ANN model
model = create_ANN(input_dim=X_train.shape[1])
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, callbacks=[early_stopping], batch_size=32, validation_split=0.1, verbose=1)

# Predict and evaluate
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Classification report
print("Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=[str(i) for i in range(6)]))

# Confusion Matrix
result = confusion_matrix(true_classes, predicted_classes)
print(result)

# Test accuracy
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Plot loss and accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Plot accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

model.save('model_emotion_classification.h5')
