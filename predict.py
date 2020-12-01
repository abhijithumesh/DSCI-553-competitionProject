from pyspark import SparkContext, SparkConf
import os
import json
import pandas as pd
import time
import math
import sys

from surprise import Dataset
from surprise import Reader
from surprise import SVD

start = time.time()

test_file = sys.argv[1]
output_file = sys.argv[2]

conf = SparkConf().setAppName("competitionProject")
sc = SparkContext(conf=conf)

start = time.time()

user_ids, business_ids, stars, dates, texts = [], [], [], [], []

with open("/home/datalib/course18137/asn251995/asn251996/data/train_review.json", "r") as fp:
    
    for line in fp:
    
        value = json.loads(line)
        user_ids += [value['user_id']]
        business_ids += [value['business_id']]
        stars += [value['stars']]
        dates += [value['date']]
        texts += [value['text']]

ratings = pd.DataFrame({"user_id": user_ids, "business_id": business_ids, "rating": stars, "date": dates, "text": texts})

usr_counts = ratings["user_id"].value_counts()
act_users = usr_counts.loc[usr_counts >= 1].index.tolist()
ratings = ratings.loc[ratings.user_id.isin(act_users)]
    
user_ids_test = []
business_ids_test = []

with open(test_file, "r") as fp:

    for line in fp:
    
        value = json.loads(line)
        user_ids_test += [value["user_id"]]
        business_ids_test += [value["business_id"]]


tst_rat = pd.DataFrame({"user_id": user_ids_test, "business_id":business_ids_test})
tst_rat['ratings'] = 0

train_set = ratings.loc[:,['user_id', 'business_id', 'rating']]
test_set = tst_rat.loc[:, ['user_id', 'business_id', 'ratings']]

reader = Reader(rating_scale = (0.0, 5.0))
train_data = Dataset.load_from_df(train_set[['user_id', 'business_id', 'rating']], reader)
test_data = Dataset.load_from_df(test_set[['user_id', 'business_id', 'ratings']], reader)

tr_abh = train_data.build_full_trainset()
tst_abh = test_data.build_full_trainset().build_testset()


svd_res = SVD(n_epochs = 27, lr_all = 0.05, reg_all = 0.1)

svd_res.fit(tr_abh)

results = svd_res.test(tst_abh)


with open(output_file, "w") as fp:

    for val in results:
        json.dump({"user_id": val[0], "business_id": val[1], "stars": val[3]}, fp)
        fp.write('\n')
        
end = time.time()
        
print("Duration:", end-start)

