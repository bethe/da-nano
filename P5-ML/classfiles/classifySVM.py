#!/usr/bin/python

def classify_svm(features_train, labels_train):
    ### import the sklearn module for GaussianNB
    from sklearn.svm import SVC

    ### create classifier
    #clf = SVC(kernel = "linear")
    clf = SVC(kernel = "rbf", C = 10000.0)

    ### fit the classifier on the training features and labels
    clf.fit(features_train, labels_train)

    return clf

def SVMAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your SVM classifier """
    clf = classify_svm(features_train, labels_train)

    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example,
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    accuracy = clf.score(features_test, labels_test)
    return accuracy