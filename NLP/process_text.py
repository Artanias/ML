import nltk
import re
import pandas as pd
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords


def compare_stemmer_and_lemmatizer(stemmer, lemmatizer, word, pos):
    """
    Print the results of stemmind and lemmitization using the passed stemmer,
    lemmatizer, word and pos (part of speech)
    """
    print("Stemmer:", stemmer.stem(word))
    print("Lemmatizer:", lemmatizer.lemmatize(word, pos))
    print()


text = ""
with open('text1.txt') as f:
    text = f.read()

# Токенезируем по предложениям
# разделяем письменный язык на предложения-компоненты.

sentences = nltk.sent_tokenize(text)
for sentence in sentences:
    print(sentence)

# Токенизация по словам, разделение предложения на слова
for sentence in sentences:
    words = nltk.word_tokenize(sentence)
    print(words)
    print()

# Лемантизация и стеммиризация
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
compare_stemmer_and_lemmatizer(stemmer,
                               lemmatizer,
                               word="seen",
                               pos=wordnet.VERB)
compare_stemmer_and_lemmatizer(stemmer,
                               lemmatizer,
                               word="drove",
                               pos=wordnet.VERB)

# стоп-слова
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
sentence = "Backgammon is one of the oldest known board games."
words = nltk.word_tokenize(sentence)
without_stop_words = [word for word in words if word not in stop_words]
print(without_stop_words)

# удаление запятых с регулярками
sentence = ""
with open('text2.txt') as f:
    sentence = f.read()
pattern = r"[^\w]"
print(re.sub(pattern, " ", sentence))

# Мешок слов
with open('text3.txt') as f:
    documents = f.read().splitlines()
print(documents)

# словарь слов
count_vectorizer = CountVectorizer()
bag_of_words = count_vectorizer.fit_transform(documents)
feature_names = count_vectorizer.get_feature_names()
print(pd.DataFrame(bag_of_words.toarray(), columns=feature_names))

# tfid
tfidf_vectorizer = TfidfVectorizer()
values = tfidf_vectorizer.fit_transform(documents)
feature_names = tfidf_vectorizer.get_feature_names()
print(pd.DataFrame(values.toarray(), columns=feature_names))
