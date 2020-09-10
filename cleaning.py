import nltk
import inflect
import unicodedata
from bs4 import BeautifulSoup
import re, string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer


def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words""" 
    return [word.lower() for word in words]

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    punct = list(string.punctuation) + ['``']
    new_words = [word for word in words
                if word not in punct]
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()            
    new_words = [p.number_to_words(word)
                if word.isdigit()
                else word
                for word in words ]
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = [word for word in words
                if word not in stopwords.words('english')]
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = [stemmer.stem(word) for word in words]
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word, pos='v') for word in words]
    return lemmas

### Combining ALL functions #########################################################################################

def normalize_text(words):
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    #words = stem_words(words)
    words = lemmatize_verbs(words)
    return words

def denoise_text(text):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    text = re.sub('\[[^]]*\]', '', text)
    return text

def text_prepare(text):
    text = denoise_text(text)
    tokens = nltk.word_tokenize(text)
    text = ' '.join([x for x in normalize_text(tokens)])
    return text