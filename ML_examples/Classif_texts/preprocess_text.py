import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import numpy as np

# load the dataset
path_df = './News_dataset.pickle'

with open(path_df, 'rb') as data:
    df = pickle.load(data)

print(df.head())
print(df.loc[1]['Content'])

# Clean text from unusing elements
df['Content_Parsed_1'] = df['Content'].str.replace('\r', ' ')
df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace('\n', ' ')
df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace('    ', ' ')
df['Content_Prsed_1'] = df['Content_Parsed_1'].str.replace('"', '')

# Lowercasing the text
df['Content_Parsed_2'] = df['Content_Parsed_1'].str.lower()

# get rid of punctuation signs
punctuation_signs = list("?:!.,;")
df['Content_Parsed_3'] = df['Content_Parsed_2']
for punct_sign in punctuation_signs:
    df['Content_Parsed_3'] = df['Content_Parsed_3'].str.replace(punct_sign, '')

# Possessive pronouns
df['Content_Parsed_4'] = df['Content_Parsed_3'].str.replace("'s", "")

# Download punkt and wordnet
nltk.download('punkt')
nltk.download('wordnet')

# Save lemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# lemmatize
nrows = len(df)
lemmatized_text_list = []

for row in range(nrows):
    lemmatized_list = []

    text = df.loc[row]['Content_Parsed_4']
    text_words = text.split(' ')

    for word in text_words:
        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos='v'))

    lemmatized_text = ' '.join(lemmatized_list)

    lemmatized_text_list.append(lemmatized_text)

df['Content_Parsed_5'] = lemmatized_text_list

# Delete stop words
nltk.download('stopwords')
stop_words = list(stopwords.words('english'))
df['Content_Parsed_6'] = df['Content_Parsed_5']
for stop_word in stop_words:
    regex_stopword = r'\b' + stop_word + r'\b'
    df['Content_Parsed_6'] = df[
                                'Content_Parsed_6'
                                ].str.replace(regex_stopword, '')


# output process text after all steps
print('Beginner text:')
print(df.loc[5]['Content'])
print('After first parse:')
print(df.loc[5]['Content_Parsed_1'])
print('After second parse:')
print(df.loc[5]['Content_Parsed_2'])
print('After third parse:')
print(df.loc[5]['Content_Parsed_3'])
print('After fourth parse:')
print(df.loc[5]['Content_Parsed_4'])
print('After fifth parse:')
print(df.loc[5]['Content_Parsed_5'])
print('After sixth parse:')
print(df.loc[5]['Content_Parsed_6'])

# delete intermediate columns
list_columns = ['File_Name',
                'Category',
                'Complete_Filename',
                'Content',
                'Content_Parsed_6']
df = df[list_columns]
df = df.rename(columns={'Content_Parsed_6': 'Content_Parsed'})
print(df.head())

# dict with label codification
category_codes = {
    'business': 0,
    'entertainment': 1,
    'politics': 2,
    'sport': 3,
    'tech': 4
}
df['Category_Code'] = df['Category']
df = df.replace({'Category_Code': category_codes})

# train - test split
X_train, X_test, y_train, y_test = train_test_split(df['Content_Parsed'],
                                                    df['Category_Code'],
                                                    test_size=0.15,
                                                    random_state=8)

# text representation
ngram_range = (1, 2)
min_df = 10
max_df = 1.
max_features = 300
tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)
features_train = tfidf.fit_transform(X_train).toarray()
labels_train = y_train
print(features_train.shape)

features_test = tfidf.transform(X_test).toarray()
labels_test = y_test
print(features_test.shape)

# see what unigrams and bigrams are most correlated with each category
for Product, category_id in sorted(category_codes.items()):
    features_chi2 = chi2(features_train, labels_train == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}' category:".format(Product))
    uni = '\n '.join(unigrams[-5:])
    bi = '\n '.join(bigrams[-2:])
    print(' . Most correlated unigrams:\n. {}'.format(uni))
    print(' . Most correlated bigrams:\n. {}'.format(bi))
    print('')

# save for the next step
with open('Pickles/X_train.pickle', 'wb') as output:
    pickle.dump(X_train, output)

with open('Pickles/X_test.pickle', 'wb') as output:
    pickle.dump(X_test, output)

with open('Pickles/y_train.pickle', 'wb') as output:
    pickle.dump(y_train, output)

with open('Pickles/y_test.pickle', 'wb') as output:
    pickle.dump(y_test, output)

with open('Pickles/df.pickle', 'wb') as output:
    pickle.dump(df, output)

with open('Pickles/features_train.pickle', 'wb') as output:
    pickle.dump(features_train, output)

with open('Pickles/labels_train.pickle', 'wb') as output:
    pickle.dump(labels_train, output)

with open('Pickles/features_test.pickle', 'wb') as output:
    pickle.dump(features_test, output)

with open('Pickles/labels_test.pickle', 'wb') as output:
    pickle.dump(labels_test, output)

with open('Pickles/tfidf.pickle', 'wb') as output:
    pickle.dump(tfidf, output)
