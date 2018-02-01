#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as pp
import numpy as np

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn import linear_model

### FUNCTIONS ###

def print_title(string):
    """Simple function to write titles in output report. Pass a string."""
    separator = "-" * 80
    print "\n"
    print separator
    print string
    print separator
    print "\n"

def feature_is_poi(feature):
    """
    Receives a feature name as argument
    and returns how many POI have all the data from them,
    that is, no NaN at all.
    """
    fees_and_poi = 0
    for name in my_dataset:
        if my_dataset[name][feature] != "NaN" and\
         my_dataset[name]["poi"] == True:
            fees_and_poi = fees_and_poi + 1
    return fees_and_poi

def find_max_name(feature):
    """
    Receives a feature name and find the person with the max value.
    """
    max_value=0
    name_max = ""
    for name in my_dataset:
        if my_dataset[name][feature] != "NaN" and my_dataset[name][feature]\
         > max_value:
            max_value = my_dataset[name][feature]
            name_max = name
    return name_max

def get_f_statistics(feature):
    """
    Stores all the values from a given feature on a numpy array.
    Then, it calculates in % the difference between the MAX and the AVG.
    The idea is to find ouliers from those percentages.
    """
    data_list = []
    for name in my_dataset:
        if my_dataset[name][feature] != "NaN":
            data_list.append(my_dataset[name][feature])
    print_title("Statistics for " + feature + ":")
    numpy_feature = np.array(data_list)
    print "MAX: " + str(int(round((np.max(numpy_feature)))))
    print "MIN: " + str(int(round((np.min(numpy_feature)))))
    print "AVG: " + str(int(round((np.average(numpy_feature)))))
    print "MAX percentage over AVG: "\
     + str(int(round((np.max(numpy_feature) * 100 )\
      / np.average(numpy_feature)))) + "%"

#def graph(feature):
#    '''
#    It shows graphs to find ouliers visually. Once the graphs has been created
#    and attached to the doc, I comment the code to avoid it from stopping
#    the execution of the rest of the code.
#    '''
#    feature_list = []
#    for name in my_dataset:
#        if my_dataset[name][feature] != "NaN":
#            feature_list.append(my_dataset[name][feature])
#    pp.plot(feature_list, 'ro')
#    pp.title(feature)
#    pp.show()

def get_metrics(clf):
    '''
    Get a classifier as parameter and use it to calculate
    precision and recall metrics.
    '''
    l_pred = clf.predict(features_test)
    print metrics.precision_score(labels_test,l_pred)
    print metrics.recall_score(labels_test,l_pred)

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
#############################################

###
### We load the dataset and also remove TOTAL and TRAVEL AGENCY
my_dataset = data_dict
#data_dict.pop( "TOTAL", 0 )
#data_dict.pop( "TOTAL", 0 )
#############################################

### Find top 5 people that have most of the NaN
print_title("Top 5 people with NaN values")
nan_dict = {}
for name in my_dataset:
    has_nan = 0
    nan_dict[name] = has_nan
    for key in my_dataset[name]:
        if my_dataset[name][key] == "NaN":
            nan_dict[name] += 1
sorted_list = sorted(nan_dict.items(), key=lambda x: x[1], reverse=True)
print sorted_list[0]
print sorted_list[1]
print sorted_list[2]
print sorted_list[3]
print sorted_list[4]
#############################################

### Person with Max payment
print "Person with max total payment is: " + find_max_name("total_payments")
# Returns TOTAL
#############################################

### From previous code we find that we have TRAVEL AGENCY and TOTAL,
### they are not people so removed from the list
data_dict.pop( "THE TRAVEL AGENCY IN THE PARK", 0 )
data_dict.pop( "TOTAL", 0 )
#############################################

### Find the number of people in our dataset
print_title("Numer of people included in the dataset (after cleanup)")
print str(len(my_dataset))
#############################################

### Extract the list of features using Walls as example
### It prints each key of the dictionary
list_of_features = []
print_title("For each person we have this list of features:")
for key in my_dataset['WALLS JR ROBERT H']:
    print "- " + key
    list_of_features.append(key)
#############################################

### How many are poi?
### Counts the number of people with poi feature = True
number_poi=0
for name in my_dataset:
    if my_dataset[name]["poi"] == True:
        number_poi = number_poi + 1
print_title("Number of people labeled as POI:")
print str(number_poi)
#############################################

### How many NaN are on each feature?
### Counts all features (keys) that are NaN
nan_dict = {}
for name in my_dataset:
    for key in my_dataset[name]:
        if nan_dict.get(key) == None:
            nan_dict[key] = 0
        if my_dataset[name][key] == "NaN":
            nan_dict[key] += 1
print_title("Number of NaN for each feature:")
for key in nan_dict:
    print key + ": " + str(nan_dict[key])
#############################################

print_title("Number of POI with director_fees:")
print str(feature_is_poi("director_fees"))
print_title("Number of POI with restricted_stock_deferred:")
print str(feature_is_poi("restricted_stock_deferred"))
print_title("Number of POI with deferral_payments:")
print str(feature_is_poi("deferral_payments"))
#############################################

### We list all features that have no data at all from POIs
print_title("List of features that are all NaN for POIs:")
for feature in list_of_features:
    if feature_is_poi(feature) == 0:
        print feature
#############################################

### The opposite of the previous one. Features that provide data
### from all POI
print_title("List of features that we have information from all POIs")
for feature in list_of_features:
    if feature_is_poi(feature) == number_poi:
        print feature
#############################################

#############################################

get_f_statistics("salary")
get_f_statistics("total_payments")
get_f_statistics("total_stock_value")
get_f_statistics("expenses")
get_f_statistics("bonus")
#############################################

#graph("total_payments")
#graph("total_stock_value")

#############################################

### Find the person with max on these two features
print_title("Person with max total payment is:")
print find_max_name("total_payments")
print_title("Person with max total_stock_value is:")
print find_max_name("total_stock_value")
#############################################

### Create new email_exchange_with_poi feature
for name in my_dataset:
    to_poi = my_dataset[name]["from_this_person_to_poi"]
    if to_poi == "NaN":
        to_poi = 0
    from_poi = my_dataset[name]["from_poi_to_this_person"]
    if from_poi == "NaN":
        from_poi = 0
    my_dataset[name]["email_exchange_with_poi"] = to_poi + from_poi

#############################################

features_list = ['poi','total_payments','total_stock_value','expenses',\
'salary','bonus','to_messages','email_exchange_with_poi']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
#############################################

### Find the best features using Lasso
lasso = linear_model.Lasso()
lasso.fit(features,labels)
print_title("Features coefficient with Lasso Regression")

print features_list
print(lasso.coef_)
#############################################

features_list = ['poi','expenses','bonus','email_exchange_with_poi']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
#############################################

from sklearn import svm, tree, grid_search, metrics

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

### Testing SVC parameters
parameters = {'C':[1, 10, 100, 1000, 10000], 'gamma': [0.0001, 0.001, 0.01,\
0.1, 1, 10, 100, 1000]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
print_title("Best SVC parameters")
clf.fit(features_train,labels_train)
print clf.best_params_

clf = svm.SVC()
clf.fit(features_train,labels_train)
print_title("SVC score with default parameters")
print clf.score(features_test,labels_test)
get_metrics(clf)

clf = svm.SVC(C = 1, gamma = 0.0001)
clf.fit(features_train,labels_train)
print_title("SVC score with recommended parameters")
print clf.score(features_test,labels_test)
get_metrics(clf)

### Testing DecisionTree parameters
parameters = {'min_samples_split' : [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 100],\
 'min_samples_leaf' : [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 100]}
svr = tree.DecisionTreeClassifier()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(features_train,labels_train)
print_title("Best Decision Tree parameters")
print clf.best_params_

clf = tree.DecisionTreeClassifier()
clf.fit(features_train,labels_train)
print_title("Decission Tree score with default parameters")
print clf.score(features_test,labels_test)
get_metrics(clf)

clf = tree.DecisionTreeClassifier(min_samples_split = 100, min_samples_leaf = 1)
clf.fit(features_train,labels_train)
print_title("Decission Tree recommended parameters")
print clf.score(features_test,labels_test)
get_metrics(clf)
#############################################

clf = tree.DecisionTreeClassifier()
clf.fit(features_train,labels_train)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
