from surprise import Dataset, evaluate
from surprise import KNNBasic
import os
import urllib.request
from azureml.core.run import Run

run = Run.get_submitted_run()

# manually downloading the file, as it requires a prompt otherwise
url='http://files.grouplens.org/datasets/movielens/ml-100k.zip'
DATASETS_DIR = os.path.expanduser('~') + '/.surprise_data/'

print("Starting")

name = 'ml-100k'
os.makedirs(DATASETS_DIR, exist_ok=True)
urllib.request.urlretrieve(url, DATASETS_DIR + 'tmp.zip')


import zipfile
with zipfile.ZipFile(DATASETS_DIR + 'tmp.zip', 'r') as tmp_zip:
    tmp_zip.extractall(DATASETS_DIR + name)

data = Dataset.load_builtin(name)
trainingSet = data.build_full_trainset()

#############################################################################################################################
# option 1: change it to item-based
# option 2: change the name (name is basically the method like cosine distance or some other metric)
# TODO: ankhokha: insert URL for future use
sim_options = {
    'name': 'cosine',
    'user_based': False # Change it to True
}
 
# option 3: use different model (like KNNWithMeans or something like that)
# TODO: ankhokha: add other model URLs
knn = KNNBasic(sim_options=sim_options)

knn.train(trainingSet)

testSet = trainingSet.build_anti_testset()
predictions = knn.test(testSet)

from collections import defaultdict
 
def get_top3_recommendations(predictions, topN = 3):
     
    top_recs = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_recs[uid].append((iid, est))
     
    for uid, user_ratings in top_recs.items():
        user_ratings.sort(key = lambda x: x[1], reverse = True)
        top_recs[uid] = user_ratings[:topN]
     
    return top_recs

import os, io
 
def read_item_names():
    """Read the u.item file from MovieLens 100-k dataset and returns a
    mapping to convert raw ids into movie names.
    """
 
    file_name = (os.path.expanduser('~') +
                 '/.surprise_data/ml-100k/ml-100k/u.item')
    rid_to_name = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
 
    return rid_to_name

'''top3_recommendations = get_top3_recommendations(predictions)
rid_to_name = read_item_names()
for uid, user_ratings in top3_recommendations.items():
    try:
        print(uid, [rid_to_name[iid] for (iid, _) in user_ratings])
    except e:
        print("Exception")'''

# write the data
import os
os.makedirs('./outputs', exist_ok=True)

from sklearn.externals import joblib

with open("model1.pkl", "wb") as file:
    joblib.dump(knn, os.path.join('./outputs/', "model1.pkl"))

#############################################################################################################################
# option 1: change it to item-based
# option 2: change the name (name is basically the method like cosine distance or some other metric)
# TODO: ankhokha: insert URL for future use
sim_options = {
    'name': 'cosine',
    'user_based': True # Change it to True
}
 
# option 3: use different model (like KNNWithMeans or something like that)
# TODO: ankhokha: add other model URLs
knn = KNNBasic(sim_options=sim_options)

knn.train(trainingSet)

testSet = trainingSet.build_anti_testset()
predictions = knn.test(testSet)

from collections import defaultdict
 
def get_top3_recommendations(predictions, topN = 3):
     
    top_recs = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_recs[uid].append((iid, est))
     
    for uid, user_ratings in top_recs.items():
        user_ratings.sort(key = lambda x: x[1], reverse = True)
        top_recs[uid] = user_ratings[:topN]
     
    return top_recs

import os, io
 
def read_item_names():
    """Read the u.item file from MovieLens 100-k dataset and returns a
    mapping to convert raw ids into movie names.
    """
 
    file_name = (os.path.expanduser('~') +
                 '/.surprise_data/ml-100k/ml-100k/u.item')
    rid_to_name = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
 
    return rid_to_name

'''top3_recommendations = get_top3_recommendations(predictions)
rid_to_name = read_item_names()
for uid, user_ratings in top3_recommendations.items():
    try:
        print(uid, [rid_to_name[iid] for (iid, _) in user_ratings])
    except e:
        print("Exception")'''

# write the data
import os
os.makedirs('./outputs', exist_ok=True)

from sklearn.externals import joblib

with open("model2.pkl", "wb") as file:
    joblib.dump(knn, os.path.join('./outputs/', "model2.pkl"))

#############################################################################################################################
# option 1: change it to item-based
# option 2: change the name (name is basically the method like cosine distance or some other metric)
# TODO: ankhokha: insert URL for future use
sim_options = {
    'name': 'cosine',
    'user_based': False # Change it to True
}
sim_options = {
    'name': 'pearson_baseline',
    'shrinkage': 0 # no shrinkage
}
 
# option 3: use different model (like KNNWithMeans or something like that)
# TODO: ankhokha: add other model URLs
knn = KNNBasic(sim_options=sim_options)

knn.train(trainingSet)

testSet = trainingSet.build_anti_testset()
predictions = knn.test(testSet)

from collections import defaultdict
 
def get_top3_recommendations(predictions, topN = 3):
     
    top_recs = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_recs[uid].append((iid, est))
     
    for uid, user_ratings in top_recs.items():
        user_ratings.sort(key = lambda x: x[1], reverse = True)
        top_recs[uid] = user_ratings[:topN]
     
    return top_recs

import os, io
 
def read_item_names():
    """Read the u.item file from MovieLens 100-k dataset and returns a
    mapping to convert raw ids into movie names.
    """
 
    file_name = (os.path.expanduser('~') +
                 '/.surprise_data/ml-100k/ml-100k/u.item')
    rid_to_name = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
 
    return rid_to_name

'''top3_recommendations = get_top3_recommendations(predictions)
rid_to_name = read_item_names()
for uid, user_ratings in top3_recommendations.items():
    try:
        print(uid, [rid_to_name[iid] for (iid, _) in user_ratings])
    except e:
        print("Exception")'''

# write the data
import os
os.makedirs('./outputs', exist_ok=True)

from sklearn.externals import joblib

with open("model3.pkl", "wb") as file:
    joblib.dump(knn, os.path.join('./outputs/', "model3.pkl"))
