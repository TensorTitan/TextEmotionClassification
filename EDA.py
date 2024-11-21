import matplotlib.pyplot as plt
from collections import Counter
from cleaned_tokenized import tokenized_cleaned

def EDA(data):
    print(data['emotion'].unique())

    emotion_counts = data['emotion'].value_counts()

    plt.figure(figsize=(8, 6))
    emotion_counts.plot(kind='bar', color='skyblue')
    plt.title('Frequency of Emotions')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

    data['word_count'] = data['review'].apply(lambda x: len(x.split()))

    average_word_count = data.groupby('emotion')['word_count'].mean()
    plt.figure(figsize=(8, 8))
    plt.title('Average word count')
    average_word_count.plot(kind='bar', color='red')
    plt.xlabel('Emotion')
    plt.ylabel('Word count')
    plt.xticks(rotation=45)
    plt.show()


def tokenization(data):
    data['tokens'] = data['review'].apply(tokenized_cleaned)

    popular_words_per_emotion = {}

    for emotion, group in data.groupby('emotion'):
        all_words = [word for tokens in group['tokens'] for word in tokens]
        word_counts = Counter(all_words)
        most_common_words = word_counts.most_common(50)
        popular_words_per_emotion[emotion] = most_common_words

    # Display the results
    for emotion, words in popular_words_per_emotion.items():
        print(f"Top 50 words for emotion '{emotion}':")
        for word, count in words:
            print(f"{word}: {count}")
        print("\n")

