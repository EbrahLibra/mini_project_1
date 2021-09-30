import numpy as np

import sklearn.datasets
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import classification_report


# Step 1: Load the corpus using load files and make sure you set the encoding to latin1. (Task 1.3)
# Get and group the data
def load_files_of_bbc(category=None):
    """
    Gets the corpus of data
    :param category: The category/class of the the instance
    :return: The corpus, the length and list of file names
    """
    files_load = sklearn.datasets.load_files('../data/BBC', description="""
     D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006.6
     """, categories=category, encoding='latin1')
    length = len(files_load.data)
    return files_load, length


# all BBC data and size
allBBCFiles, allBBCDataSize = load_files_of_bbc()

# #BBC business Data and size
businessFiles, businessDataSize = load_files_of_bbc('business')

# #BBC entertainment Data and size
entertainmentFiles, entertainmentDataSize = load_files_of_bbc('entertainment')

# #BBC politics Data and size
politicsFiles, politicsDataSize = load_files_of_bbc('politics')

# #BBC sport Data and size
sportFiles, sportDataSize = load_files_of_bbc('sport')

# #BBC tech Data
techFiles, techDataSize = load_files_of_bbc('tech')

# Step 2: Plot the distribution of the instances in each class
# and save the graphic in a file called BBC-distribution.pdf. (Task 1.2)

# Create Dataframe
row_name = 'Records Count'
allBBC_DF = pd.DataFrame({
    'Business': businessDataSize,
    'Entertainment': entertainmentDataSize,
    'Politics': politicsDataSize,
    'Sport': sportDataSize,
    'Tech': techDataSize
},
    index=[row_name]
)

print(allBBC_DF)

# Plot the distribution of the instances in each class
sp = allBBC_DF.loc[row_name].plot(by=allBBCFiles.target_names, title=row_name, figsize=(12, 6),
                                  kind='barh')

# Save the graphic in a file called BBC-distribution.pdf
fig = sp.get_figure()
fig.savefig('../out/BBC-distribution.pdf')

# Step 3: Pre-process the dataset to have the features ready
# to be used by a multinomial Naive Bayes classifier. (Task 1.4)

vectorizer = CountVectorizer()


def get_matrix_and_vocabulary(data):
    return vectorizer.fit_transform(data).toarray(), {k: v for k, v in
                                                      sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1])}


# Extract matrices
# creating X and Y for train test split (merging all directories and assigning each entry its proper label

allBBC_Matrix, allBBC_Vocab = get_matrix_and_vocabulary(allBBCFiles.data)
businessMatrix, businessVocab = get_matrix_and_vocabulary(businessFiles.data)
entertainmentMatrix, entertainmentVocab = get_matrix_and_vocabulary(entertainmentFiles.data)
politicsMatrix, politicsVocab = get_matrix_and_vocabulary(politicsFiles.data)
sportMatrix, sportVocab = get_matrix_and_vocabulary(sportFiles.data)
techMatrix, techVocab = get_matrix_and_vocabulary(techFiles.data)

# Step 4: Split the dataset into 80% for training and 20% for testing.
# For this, you must use train test split with the parameter random state set to None. (Task 1.5)

X_train, X_test, y_train, y_test = train_test_split(allBBC_Matrix, allBBCFiles.target, test_size=0.20)

# # Step 5: Train a multinomial Naive Bayes Classifier (naive bayes.MultinomialNB) on the training set using
# # the default parameters and evaluate it on the test set. (Task 1.6)

# # Prepare the classifier
i = len(allBBCFiles.target_names)
model = MultinomialNB(class_prior=[
    businessDataSize / allBBCDataSize,
    entertainmentDataSize / allBBCDataSize,
    politicsDataSize / allBBCDataSize,
    sportDataSize / allBBCDataSize,
    techDataSize / allBBCDataSize
],
    alpha=0.9
)

# training
model.fit(X_train, y_train)

# predictions
predictions = model.predict(X_test)

# # Step 6: In a file called bbc-performance.txt, save the following information: (to make it easier for the TAs, make
# # sure that your output for each sub-question below is clearly marked in your output file, using the headings
# # (a), (b) . . . ) (Task 1.7)

with open('../out/bbc-performance.txt', 'a') as f:
    f.write('\n' + '-' * 70)
    f.write('\na)\n\t' + '*' * 10 + ' Multi-nominalNB default values, try 4 (smoothing={0:3.5})'.format(model.alpha) + '*' * 10)
    f.write('\nb)\tConfusion Matrix:')
    f.write('\n\n\t' + np.array2string(metrics.confusion_matrix(y_test, predictions), prefix='\t\t\t'))
    f.write('\n\nc-d)\n\t' + classification_report(y_test, predictions, target_names=allBBCFiles.target_names) + '\n')
    f.write('e)\tThe prior probability of each class:\n')
    for i, pp in enumerate(model.class_prior):
        f.write('\n\t' + allBBCFiles.target_names[i] + ':\t{0:1.2}'.format(pp))
    f.write('\n\nf)\tThe size of the vocabulary:\t{0}\n'.format(len(allBBC_Vocab)))
    f.write('\ng)')
    f.write('\n\tThe number of word-tokens in business:\t\t\t\t{0}'.format(businessMatrix.sum()))
    f.write('\n\tThe number of word-tokens in entertainment:\t\t\t{0}'.format(entertainmentMatrix.sum()))
    f.write('\n\tThe number of word-tokens in politics:\t\t\t\t{0}'.format(politicsMatrix.sum()))
    f.write('\n\tThe number of word-tokens in sport:\t\t\t\t\t{0}'.format(sportMatrix.sum()))
    f.write('\n\tThe number of word-tokens in tech:\t\t\t\t\t{0}'.format(techMatrix.sum()))
    f.write('\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t------')
    f.write('\nh)\tThe number of word-tokens in the entire corpus:\t\t{0}'.format(allBBC_Matrix.sum()))
    f.write('\nk)\tMy favourite words are')
    f.write('\n\t\t"french" with log-prob:\t\t\t{0}'.format(model.feature_log_prob_[:, allBBC_Vocab['french']].sum()))
    f.write('\n\t\t"freedom" with log-prob:\t\t{0}'.format(model.feature_log_prob_[:, allBBC_Vocab['freedom']].sum()))
