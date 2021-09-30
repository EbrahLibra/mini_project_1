import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

# Load dataset
drug_dataset = pd.read_csv('./drug200.csv')

# Show head
drug_dataset.head()

sp = drug_dataset['Drug'].value_counts().plot(title='Drug instance count', figsize=(18, 6),
                                              kind='barh')

plt.savefig('./out/drug-distribution.pdf')

# cleanup
cleanup_enums = {
    "Sex": { "F": 0, "M": 1 },
    "BP": {"LOW": 0, "NORMAL": 1, "HIGH": 2},
    "Cholesterol": {"LOW": 0, "NORMAL": 1, "HIGH": 2},
#     "Drug": {"drugA": 0, "drugB": 1, "drugC": 3, "drugX": 4, "drugY": 5}
}


drug_dataset = drug_dataset.replace(cleanup_enums)

print(drug_dataset.head())

# train/test split
X = drug_dataset.loc[:, drug_dataset.columns != 'Drug']
Y = drug_dataset['Drug']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

# 6 a)
gnb = GaussianNB()

gnb_prediction = gnb.fit(X_train, y_train).predict(X_test)

# 6 b)
dtc = DecisionTreeClassifier()

dtc_prediction = gnb.fit(X_train, y_train).predict(X_test)

# 6 c)


# 6 d)
per = Perceptron()

per_prediction = per.fit(X_train, y_train).predict(X_test)

# 6 e)
base_mlp = MLPClassifier(hidden_layer_sizes=100)

base_mlp_prediction = base_mlp.fit(X_train, y_train).predict(X_test)