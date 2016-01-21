#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

### Reduce training set to 1% to speed up algorhithm
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]


#########################################################
### your code goes here ###

#########################################################

def SVMPredict(features_train, labels_train, features_test):
    ### import the sklearn module for GaussianNB
    from sklearn.svm import SVC

    ### create classifier
    #clf = SVC(kernel = "linear")
    clf = SVC(kernel = "rbf", C = 10000.0)

    ### fit the classifier on the training features and labels
    t0 = time()
    clf.fit(features_train, labels_train)
    print "training time:", round(time()-t0, 3), "s"

    ### use the trained classifier to predict labels for the test features
    t1 = time()
    pred = clf.predict(features_test)
    print "prediction time:", round(time()-t1, 3), "s"
    return clf, pred


def SVMAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your SVM classifier """
    clf, pred = SVMPredict(features_train, labels_train, features_test)

    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example,
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    accuracy = clf.score(features_test, labels_test)
    return accuracy

#print SVMAccuracy(features_train, labels_train, features_test, labels_test)

### Get predictions for items 10, 26 and 50
#print "No 10:", SVMPredict(features_train, labels_train, features_test[10])[1]
#print "No 26:", SVMPredict(features_train, labels_train, features_test[26])[1]
#print "No 50:", SVMPredict(features_train, labels_train, features_test[50])[1]

### Get total classifications
print sum(SVMPredict(features_train, labels_train, features_test)[1])
