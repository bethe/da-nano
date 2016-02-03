#!/usr/bin/python

def classify_rf(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
            
    ### your code goes here!
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators = 1000, min_samples_split = 50)
    clf = clf.fit(features_train, labels_train)   
    return clf


def RFAccuracy(features_train, labels_train, features_test, labels_test):

    ### fit the classifier on the training features and labels
    clf = classify_rf(features_train, labels_train)

    ### use the trained classifier to predict labels for the test features
    pred = clf.predict(features_test)

    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example, 
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    accuracy = clf.score(features_test, labels_test)
    return accuracy