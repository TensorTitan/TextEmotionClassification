from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

removed_words = ['feel', 'feeling', 'really', 'im',
                 'like', 'know', 'want', 'really', 'little', 'even', 'way', 'things', 'could', 'bit',
                 'bit', 'still', 'think', 'would', 'get', 'time', 'people', 'dont', 'going', 'someone', 'day', 'today',
                 'ive', 'less', 'always', 'also', 'see', 'got', 'ever', 'many', 'much', 'one', 'back']
def tokenized_cleaned(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in punctuation and word not in stop_words and word not in removed_words]
    return tokens
