from pyspark import SparkContext, SparkConf
import os
import json

pp_coeff = {}
def predict(user, business):

    pearson_sim_weights = {}
    
    if(business_user_train.get(business)):
        corated_users = set(business_user_train[business])
    else:
        corated_users = set()
        
    if(user_business_train.get(user)):
        user_business_interests = set(user_business_train[user])
    else:
        return ((user,business),1.0)   
    
    
    if len(corated_users) >= 50:
    
        for bn in user_business_interests:
            c_users = set()
            bn_users = set(business_user_train[bn])
            
            c_users = corated_users.intersection(bn_users)
            
            if len(c_users >= 2):
                c_users = set(list(c_users)[:5])
                
            if (business,bn) not in pearson_sim_weights and len(c_users) >= 2:
            
                b1_ratings = 0 
                b2_ratings = 0
                pearson_W_ij = 0
                
                
                for usr in c_users:
                    b1_ratings += rdd_train[(usr, business)]
                    b2_ratings += rdd_train[(usr, bn)]
                    
                b1_avg = b1_ratings / len(c_users)
                b2_avg = b2_ratings / len(c_users)
                
                W_num = 0
                W_den = 0
                b1_den = 0
                b2_den = 0
                
                for usr in c_users:
                
                    b1_rating = rdd_train[(usr,business)] - b1_avg
                    b2_rating = rdd_train[(usr,bn)] - b2_avg
                    
                    W_num = W_num + b1_rating * b2_rating
                    
                    b1_den += b1_rating ** 2
                    b2_den += b2_rating ** 2
                    
                W_den = math.sqrt(b1_den) * math.sqrt(b2_den)
                
                if W_den != 0 and W_num != 0:
                    pearson_W_ij = abs(W_num/W_den)
                    
                pearson_sim_weights[(business,bn)] = pearson_W_ij
                pp_coeff[(business,bn)] = pearson_sim_weights[(business,bn)]
            
            
            if (business, bn) in pp_coeff:
                return ((business, bn), pp_coeff[(business, bn)])
            else:
                return ((business, bn), 0.0)


conf = SparkConf().setAppName("competitionProject")
sc = SparkContext(conf=conf)

rdd = sc.textFile('/home/datalib/course18137/asn251995/asn251996/data/train_review.json')
js = rdd.map(lambda x: json.loads(x))
rdd1 = js.map(lambda x: ((x['user_id'], x['business_id']), x['stars']))
rdd_train = rdd1.collectAsMap()

#user1: (b1, b2, b3......), user2: (b3, b4, b6....)
user_business_train = rdd1.map(lambda x: (x[0][0], x[0][1])).groupByKey().mapValues(lambda x: set(x)).sortByKey().collectAsMap()
#business1: (u1, u2...), business2: (u3, u4...)
business_user_train = rdd1.map(lambda x: (x[0][1], x[0][0])).groupByKey().mapValues(lambda x: set(x)).sortByKey().collectAsMap()

rdd_training = rdd1.map(lambda x: (x[0][0], x[0][1])).sortByKey()

predictions = rdd_training.map(lambda x: predict(x[0], x[1]))
similarity = predictions.filter(lambda x: x is not None).map(lambda x: {"b1": x[0][0], "b2": x[0][1], "sim": x[1]}).collect()

fp = open('model_file.model', 'w')

for i in similarity:
    json.dump(i, fp)
    fp.write('\n')
    
fp.close()