import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
drug_dataset = pd.read_csv('../data/drug200.csv')

# Show head
head = drug_dataset.head()
print(head)

sp = drug_dataset['Drug'].value_counts().plot(title='Drug instance count', figsize=(18, 6),
                                              kind='barh')

plt.savefig('../out/drug-distribution.pdf')

# cleanup
cleanup_enums = {
    "Sex": {"F": 0, "M": 1},
    "BP": {"LOW": 0, "NORMAL": 1, "HIGH": 2},
    "Cholesterol": {"LOW": 0, "NORMAL": 1, "HIGH": 2},
    #     "Drug": {"drugA": 0, "drugB": 1, "drugC": 3, "drugX": 4, "drugY": 5}
}

drug_dataset = drug_dataset.replace(cleanup_enums)

print('\n', drug_dataset.head())

# train/test split
X = drug_dataset.loc[:, drug_dataset.columns != 'Drug']
Y = drug_dataset['Drug']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
predictions = {}

# 6 a)
gnb = GaussianNB()

gnb_prediction = gnb.fit(X_train, y_train).predict(X_test)
predictions.update({'Gaussian Naive Bayes Classifier': gnb_prediction})
# 6 b)
dtc = DecisionTreeClassifier()

dtc_prediction = dtc.fit(X_train, y_train).predict(X_test)
predictions.update({'Decision Tree Classifier': dtc_prediction})
# 6 c)
param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [5, 10], 'min_samples_split': [2, 4, 6]}
cv = GridSearchCV(dtc, param_grid=param_grid)

cv_prediction = cv.fit(X_train, y_train).predict(X_test)
predictions.update({'GridSearchCV Classifier': cv_prediction})
# 6 d)
per = Perceptron()

per_prediction = per.fit(X_train, y_train).predict(X_test)
predictions.update({'Perceptron Classifier': per_prediction})
# 6 e)
base_mlp = MLPClassifier(activation='logistic', hidden_layer_sizes=100, solver='sgd', max_iter=2500)

base_mlp_prediction = base_mlp.fit(X_train, y_train).predict(X_test)
predictions.update({'Multi-Layered Perceptron (Base-MLP) Classifier': (base_mlp.get_params(False), base_mlp_prediction)})
# 6  f)
activation_functions = ['identity', 'logistic', 'relu', 'tanh']
solvers = ['sgd', 'adam']
hidden_layer_sizes = [(30, 50), (10, 10, 10)]


def generate_multi_layered_predictions(afs, ss, hls_s):
    predictions = {}
    for af in afs:
        for s in ss:
            for hls in hls_s:
                mlp_classifier = MLPClassifier(activation=af, hidden_layer_sizes=hls, solver=s, max_iter=4000)
                predictions.update(
                    {
                        str(mlp_classifier.get_params(False)): mlp_classifier.fit(X_train, y_train).predict(X_test)
                    }
                )
    return predictions


top_mlp_predictions = generate_multi_layered_predictions(activation_functions, solvers, hidden_layer_sizes)
predictions.update({'Multi-Layered Perceptron (Top-MLP) Classifier': top_mlp_predictions})


def generate_drug_performance(a, ps):
    with open('../out/drug-performance.txt', 'w') as f:
        f.write('\n' + '*' * 30 + 'Try ' + str(a) + '*' * 30)
        for i, (k, p) in enumerate(ps.items()):
            f.write('\n ' + chr(i + 97) + ') ' + k)
            if k != 'Multi-Layered Perceptron (Top-MLP) Classifier' and k != 'Multi-Layered Perceptron (Base-MLP) Classifier':
                f.write('\n\t 1) Confusion matrix:\n\n\t' + np.array2string(confusion_matrix(y_test, p), prefix='\t\t\t'))
                f.write('\n\t 2) Confusion matrix:\n\n' + classification_report(y_test, p, zero_division=0))
            elif k == 'Multi-Layered Perceptron (Base-MLP) Classifier':
                f.write('\nHyper-parameters:\n\t' + str(p[0]))
                f.write('\n\t 1)\tConfusion matrix:\n\n\t' + np.array2string(confusion_matrix(y_test, p[1]), prefix='\t\t\t'))
                f.write('\n\t 2)\tConfusion matrix:\n\n' + classification_report(y_test, p[1], zero_division=0))
            else:
                f.write('\n')
                for i, (p1, p2) in enumerate(p.items()):
                    f.write('\n' + '*' * 10 + ' Top-MLP (' + str(i + 1) + ') ' + '*' * 10)
                    f.write('\n\tHyper-parameters:\n\t' + p1)
                    f.write('\n\t 1)\tConfusion matrix:\n\n\t' + np.array2string(confusion_matrix(y_test, p2), prefix='\t\t\t'))
                    f.write('\n\t 2)\tConfusion matrix:\n\n' + classification_report(y_test, p2, zero_division=0))


generate_drug_performance(1, predictions)
generate_drug_performance(2, predictions)
