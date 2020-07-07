import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


def prepare_data():
    reviews_train = load_files('aclImdb/train')
    text_train, y_train = reviews_train.data, reviews_train.target

    new_texts = []
    new_y = []
    for i in range(len(y_train)):
        if y_train[i] == 0 or y_train[i] == 1:
            new_texts.append(text_train[i])
            new_y.append(y_train[i])
    text_train = new_texts
    y_train = new_y

    reviews_test = load_files('aclImdb/test')
    text_test, y_test = reviews_test.data, reviews_test.target
    print('prepare done')

    return text_train, y_train, text_test, y_test


def preprocess_data(text_train, text_test):
    cv = CountVectorizer()
    cv.fit(text_train)
    X_train = cv.transform(text_train)
    X_test = cv.transform(text_test)
    print('preprocess done')
    return X_train, X_test


def LR_model(X_train, y_train, X_test, y_test):
    logit = LogisticRegression(n_jobs=-1, random_state=7, max_iter=10000)
    logit.fit(X_train, y_train)
    print('LR done')
    return logit.score(X_train, y_train), logit.score(X_test, y_test)


def pipe_model(text_train, y_train, text_test, y_test):
    text_pipe_logit = make_pipeline(CountVectorizer(),
                                    LogisticRegression(n_jobs=-1,
                                                       random_state=7,
                                                       max_iter=10000))
    text_pipe_logit.fit(text_train, y_train)
    print('pipe done')
    return text_pipe_logit, (text_pipe_logit.score(text_train, y_train),
                             text_pipe_logit.score(text_test, y_test))


def grid_logit(text_pipe_logit, text_train, y_train, text_test, y_test):
    param_grid_logit = {'logisticregression__C': np.logspace(-5, 0, 6)}
    grid_logit = GridSearchCV(text_pipe_logit,
                              param_grid_logit,
                              cv=3,
                              n_jobs=-1)
    grid_logit.fit(text_train, y_train)
    print('grid done')
    return (grid_logit.score(text_train, y_train),
            grid_logit.score(text_test, y_test))


def RF_model(X_train, y_train, X_test, y_test):
    forest = RandomForestClassifier(n_estimators=200,
                                    n_jobs=-1,
                                    random_state=17)
    forest.fit(X_train, y_train)
    print('RF done')
    return forest.score(X_train, y_train), forest.score(X_test, y_test)


def plot_histogramm(Accuracy):
    use, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].bar(['LR', 'Pipe', 'Grid', 'RF'], [el[0] for el in Accuracy],
              color='blue', align='center')
    ax[0].set_xlabel('Models')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_title('Train')
    ax[1].bar(['LR', 'Pipe', 'Grid', 'RF'], [el[1] for el in Accuracy],
              color='blue', align='center')
    ax[1].set_xlabel('Models')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Test')
    plt.savefig('Accuracy.png')
    plt.show()


if __name__ == '__main__':
    Accuracy = []
    text_train, y_train, text_test, y_test = prepare_data()
    X_train, X_test = preprocess_data(text_train, text_test)
    Accuracy.append(LR_model(X_train, y_train, X_test, y_test))
    model, accur = pipe_model(text_train, y_train, text_test, y_test)
    Accuracy.append(accur)
    Accuracy.append(grid_logit(model, text_train, y_train, text_test, y_test))
    Accuracy.append(RF_model(X_train, y_train, X_test, y_test))
    plot_histogramm(Accuracy)
