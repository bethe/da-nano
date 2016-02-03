# coding: utf-8
# %load "../ud120-files/k_means/k_means_cluster.py"
#!/usr/bin/python 

import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../ud120-files/tools/")
from feature_format import featureFormat, targetFeatureSplit


def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../ud120-files/final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)


### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )


### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
print "Basic Scatter plot:"
for f1, f2 in finance_features:
    plt.scatter( f1, f2 )
plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
from sklearn import cluster
features_list = ["poi", feature_1, feature_2]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )
clf = KMeans(n_clusters=2)
pred = clf.fit_predict( finance_features )
print "2 Clusters; 2 Features (salary, exercised_stock_options)"
Draw(pred, finance_features, poi, name="clusters_before_scaling2.pdf", f1_name=feature_1, f2_name=feature_2)


### Q2: 3 features:
feature_3 = "total_payments"    # added for Q2
features_list3 = ["poi", feature_1, feature_2,feature_3]
data3 = featureFormat(data_dict, features_list3 )
poi, finance_features = targetFeatureSplit( data3 )
clf = KMeans(n_clusters=2)
pred = clf.fit_predict( finance_features )
print "2 Clusters; 3 Features (added total_payments)"
Draw(pred, finance_features, poi, name="clusters_before_scaling3.pdf", f1_name=feature_1, f2_name=feature_2)


### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
#try:
#    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
#except NameError:
#    print "no predictions object named pred found, no clusters to plot"
#

get_ipython().magic(u'save "../ud120-files/k_means/k_means_cluster.py" 33')
