#!/usr/bin/python

def classify_dt(features_train, labels_train):
    
    ### your code goes here--should return a trained decision tree classifer
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(min_samples_split = 50)   # default: min_samples_split = 2
    clf = clf.fit(features_train, labels_train)
    return clf

def DTAccuracy(features_train, labels_train, features_test, labels_test):

    clf = classify_dt(features_train, labels_train)

    ### use the trained classifier to predict labels for the test features
    pred = clf.predict(features_test)


    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example, 
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    accuracy = clf.score(features_test, labels_test)
    return accuracy