# --- Step 1. Import relevant packages --- #
# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


def performance(classifier, X_test, y_test, predicted):
    classifier_name = str(classifier).replace('()','')
    if 'KN' in classifier_name:
        classifier_name = 'KNN'
    elif 'SGD' in classifier_name:
        classifier_name = 'SGD'
    else:
        classifier_name = 'DT'

#     accuracy = round(metrics.accuracy_score(y_test, predicted),3)
    accuracy = round(classifier.score(X_test, y_test),5)
    recall = round(metrics.recall_score(y_test, predicted, labels = [0,1,2,3,4,5,6,7,8,9], average = 'weighted'),3)
    print(f'{classifier_name}\tAccuracy: {accuracy}\tRecall: {recall}')

# ----------------------------MAIN---------------------------------#

# --- Start --- #
print('\n\033[1mIMPROVEMENT\033[0m')
# The digits dataset
digits = datasets.load_digits()

# --- Step 2. Load the images using sklearn’s load_digits(). --- #
# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
data_size = len(digits.data)
print('Total images:',data_size)


# --- Step 3. Split the images using sklearn’s train_test_split() --- #
# Split data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.25, shuffle=False)

# ----------------------- KNeighborsClassifier ---------------------- #

classifier_KNN = KNeighborsClassifier(n_neighbors = 1)
classifier_KNN.fit(X_train, y_train)
predicted_KNN = classifier_KNN.predict(X_test)
performance(classifier_KNN, X_test,  y_test, predicted_KNN)


# ------------------------ SGDClassifier ------------------------------- #

classifier = SGDClassifier()
classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)
performance(classifier, X_test, y_test, predicted)

# ------------------------ DecisionTreeClassifier ----------------------- #

# Create a classifier: a support vector classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)
performance(classifier, X_test, y_test, predicted)


# -------------- Print confusion matrix for best classifier ------------- #
disp = metrics.plot_confusion_matrix(classifier_KNN, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
# print("Confusion matrix:\n%s" % disp.confusion_matrix)
plt.show()
