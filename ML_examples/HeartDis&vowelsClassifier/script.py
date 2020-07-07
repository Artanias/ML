import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def prepare_data_SAHD():
    heart = pd.read_csv('data.csv', sep=',', header=0)
    y = heart.iloc[:, 10]
    x = heart.iloc[:, 1:10]
    for i in range(x['famhist'].size):
        if x['famhist'][i] == 'Present':
            x.loc[i, 'famhist'] = 0
        else:
            x.loc[i, 'famhist'] = 1
    return x, y


def prepare_data_V():
    vowel_train = pd.read_csv('vowel.train.csv', sep=',', header=0)
    vowel_test = pd.read_csv('vowel.test.csv', sep=',', header=0)
    y_tr = vowel_train.iloc[:, 1]
    x_tr = vowel_train.iloc[:, 2:]
    y_test = vowel_test.iloc[:, 1]
    x_test = vowel_test.iloc[:, 2:]
    return x_tr, y_tr, x_test, y_test


def LR_model_SAHD(x, y):
    LR = LogisticRegression(random_state=0, solver='lbfgs',
                            multi_class='ovr', max_iter=1000).fit(x, y)
    return LR.score(x, y)


def LR_model_V(x_tr, y_tr, x_test, y_test):
    LR = LogisticRegression(random_state=0,
                            solver='lbfgs',
                            multi_class='multinomial',
                            max_iter=1000)
    LR.fit(x_tr, y_tr)
    return LR.score(x_tr, y_tr), LR.score(x_test, y_test)


def SVM_model_SAHD(x, y):
    SVM = svm.LinearSVC(max_iter=10000000)
    SVM.fit(x, y)
    return SVM.score(x, y)


def SVM_model_V(x_tr, y_tr, x_test, y_test):
    SVM = svm.SVC(decision_function_shape='ovo')
    SVM.fit(x_tr, y_tr)
    return SVM.score(x_tr, y_tr), SVM.score(x_test, y_test)


def RF_model_SAHD(x, y):
    RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    RF.fit(x, y)
    return RF.score(x, y)


def RF_model_V(x_tr, y_tr, x_test, y_test):
    RF = RandomForestClassifier(n_estimators=1000,
                                max_depth=10,
                                random_state=0)
    RF.fit(x_tr, y_tr)
    return RF.score(x_tr, y_tr), RF.score(x_test, y_test)


def NN_model_SAHD(x, y):
    NN = MLPClassifier(solver='lbfgs', alpha=0.00001,
                       hidden_layer_sizes=(5, 2), random_state=1)
    NN.fit(x, y)
    return NN.score(x, y)


def NN_model_V(x_tr, y_tr, x_test, y_test):
    NN = MLPClassifier(solver='lbfgs',
                       alpha=0.00001,
                       hidden_layer_sizes=(150, 10),
                       random_state=1,
                       max_iter=1000)
    NN.fit(x_tr, y_tr)
    return NN.score(x_tr, y_tr), NN.score(x_test, y_test)


def plot_histogramm(bin_acc, mc_acc):
    use, ax = plt.subplots(1, 3, figsize=(12, 5))
    ax[0].bar(['LR', 'SVM', 'RF', 'NN'], bin_acc, color='blue', align='center')
    ax[0].set_xlabel('Models')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_title('Bin Classif.')
    ax[1].bar(['LR', 'SVM', 'RF', 'NN'], [el[0] for el in mc_acc],
              color='blue', align='center')
    ax[1].set_xlabel('Models')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('MC Classif train.')
    ax[2].bar(['LR', 'SVM', 'RF', 'NN'], [el[1] for el in mc_acc],
              color='blue', align='center')
    ax[2].set_xlabel('Models')
    ax[2].set_ylabel('Accuracy')
    ax[2].set_title('MC Classif test.')
    plt.savefig('Accuracy.png')
    plt.show()


if __name__ == '__main__':
    x, y = prepare_data_SAHD()
    bin_acc = []
    print('Binary Classification example.')
    bin_acc.append(LR_model_SAHD(x, y))
    print(f'Logistic Regression accuracy: { bin_acc[0] }')
    bin_acc.append(SVM_model_SAHD(x, y))
    print(f'SVM accuracy: { bin_acc[1] }')
    bin_acc.append(RF_model_SAHD(x, y))
    print(f'RF accuracy: { bin_acc[2] }')
    bin_acc.append(NN_model_SAHD(x, y))
    print(f'NN accuracy: { bin_acc[3] }')

    mc_acc = []
    x_tr, y_tr, x_test, y_test = prepare_data_V()
    print('Multi-Class Classification example.')
    mc_acc.append(LR_model_V(x_tr, y_tr, x_test, y_test))
    print(f'Logistic Regression accuracy: { mc_acc[0] }')
    mc_acc.append(SVM_model_V(x_tr, y_tr, x_test, y_test))
    print(f'SVM accuracy: { mc_acc[1] }')
    mc_acc.append(RF_model_V(x_tr, y_tr, x_test, y_test))
    print(f'RF accuracy: { mc_acc[2] }')
    mc_acc.append(NN_model_V(x_tr, y_tr, x_test, y_test))
    print(f'NN accuracy: { mc_acc[3] }')
    plot_histogramm(bin_acc, mc_acc)
