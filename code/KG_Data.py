 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import random
import pandas as pd
import pdb

import numpy as np
from numpy.random import binomial
from os import path

class KG_Data:
    def __init__(self, data_dir, negative_sampling):
        self.data_dir = data_dir
        self.negative_sampling = negative_sampling

        self.entity_to_id = {}
        self.id_to_entity = {}
        self.relation_to_id = {}
        self.id_to_relation = {}

        self.triplets_train_pool = set()  # {(id_head, id_relation, id_tail), ...}
        self.triplets_train = []  # [(id_head, id_relation, id_tail), ...]
        self.triplets_validate = []
        self.triplets_test = []

        self.num_entity = 0
        self.num_relation = 0
        self.num_triplets_train = 0
        self.num_triplets_validate = 0
        self.num_triplets_test = 0

        # for reducing false negative labels
        self.relation_dist = {}  # {relation, (head_per_tail, tail_per_head)}
        
        # for bern sampling method, we pre generate a probability list
        self.ht_percent = np.empty(0)
        
        self.triplets_total = []
        self.triplets_total_pool = set()
        self.num_triplets_total = 0
        
        self.load_data()

    def load_data(self):
        # read the entity_to_id file
        print('loading entities...')
        entity_to_id_df = pd.read_csv(path.join(self.data_dir, 'entity2id.txt'), sep='\t', header = None, names = ['entity', 'id'])
        self.entity_to_id = dict(zip(entity_to_id_df['entity'], entity_to_id_df['id']))
        self.id_to_entity = dict(zip(entity_to_id_df['id'], entity_to_id_df['entity']))
        self.num_entity = len(self.entity_to_id)
        print('got {} entities'.format(self.num_entity))

        # read the relation_to_id file
        print('loading relations...')
        relation_to_id_df = pd.read_csv(path.join(self.data_dir, 'relation2id.txt'), sep='\t', header = None, names = ['relation', 'id'])
        self.relation_to_id = dict(zip(relation_to_id_df['relation'], relation_to_id_df['id']))
        self.id_to_relation = dict(zip(relation_to_id_df['id'], relation_to_id_df['relation']))
        self.num_relation = len(self.relation_to_id)
        print('got {} relations'.format(self.num_relation))

        # read the train file
        print('loading train triplets...')
        triplets_train_df = pd.read_csv(path.join(self.data_dir, 'train.txt'), sep='\t', header = None, names = ['head', 'tail', 'relation'])
        self.triplets_train = list(zip(
            [self.entity_to_id[head] for head in triplets_train_df['head']],
            [self.relation_to_id[relation] for relation in triplets_train_df['relation']],
            [self.entity_to_id[tail] for tail in triplets_train_df['tail']]
        ))
        
        self.num_triplets_train = len(self.triplets_train)
        print('got {} triplets from training set'.format(self.num_triplets_train))
        # construct the train triplets pool
        self.triplets_train_pool = set(self.triplets_train)
        
        # add one list to store the 
        # pdb.set_trace()
        
        if self.negative_sampling == 'bern':
            self.set_bernoulli(triplets_train_df)
        else:
            print('do not need to calculate hpt & tph...')

        # read the validate file
        print('loading validate triplets...')
        triplets_validate_df = pd.read_csv(path.join(self.data_dir, 'valid.txt'), sep='\t', header = None, names = ['head', 'tail', 'relation'])
        self.triplets_validate = list(zip(
            [self.entity_to_id[head] for head in triplets_validate_df['head']],
            [self.relation_to_id[relation] for relation in triplets_validate_df['relation']],
            [self.entity_to_id[tail] for tail in triplets_validate_df['tail']]
        ))
        self.num_triplets_validate = len(self.triplets_validate)
        print('got {} triplets from validation set'.format(self.num_triplets_validate))

        # read the test file
        print('loading test triplets...')
        triplets_test_df = pd.read_csv(path.join(self.data_dir, 'test.txt'), sep='\t', header = None, names = ['head', 'tail', 'relation'])
        self.triplets_test = list(zip(
            [self.entity_to_id[head] for head in triplets_test_df['head']],
            [self.relation_to_id[relation] for relation in triplets_test_df['relation']],
            [self.entity_to_id[tail] for tail in triplets_test_df['tail']]
        ))
        self.num_triplets_test = len(self.triplets_test)
        print('got {} triplets from test set'.format(self.num_triplets_test))
        
        self.triplets_total = self.triplets_train + self.triplets_validate + self.triplets_test
        self.triplets_total_pool = set(self.triplets_total)
        
        self.num_triplets_total = len(self.triplets_total)
        
        
        
        
    def set_bernoulli(self, triplets_train_df):
        print('calculating hpt & tph for reducing negative false labels...')
        grouped_relation = triplets_train_df.groupby('relation', as_index=False)
        # calculate head_per_tail and tail_per_head after group by relation
        n_to_one = grouped_relation.agg({
            'head': lambda heads: heads.count(),
            'tail': lambda tails: tails.nunique()
        })
        # one_to_n dataframe, columns = ['relation', 'head', 'tail']
        one_to_n = grouped_relation.agg({
            'head': lambda heads: heads.nunique(),
            'tail': lambda tails: tails.count()
        })
        relation_dist_df = pd.DataFrame({
            'relation': n_to_one['relation'],
            'head_per_tail': n_to_one['head'] / n_to_one['tail'],  # element-wise division
            'tail_per_head': one_to_n['tail'] / one_to_n['head']
        })

        self.relation_dist = dict(zip(
            [self.relation_to_id[relation] for relation in relation_dist_df['relation']],
            zip(
                relation_dist_df['head_per_tail'],
                relation_dist_df['tail_per_head']
            )
        ))
        
        # calculate bern probabilities
        self.ht_percent = np.zeros(self.num_relation)
        for id_relation in self.relation_to_id.values():
            hpt, tph = self.relation_dist[id_relation]
            head_prob = tph/(tph+hpt)
            self.ht_percent[id_relation] = head_prob
              
    def next_batch_train_parallel(self, batch_size):
        # construct positive batch
        batch_positive = random.sample(self.triplets_train, batch_size)
        
        # construct negative batch
        batch_negative = batch_positive
        
        batch_len = len(batch_negative)
        
        batch_positive_matrix = np.asarray(batch_positive)
        batch_negative_matrix = np.asarray(batch_negative)
        
        if self.negative_sampling == 'unif':
            head_prob = binomial(1, 0.5, size = batch_len)
        else:
            
            ht_per = self.ht_percent[batch_negative_matrix[:,1]]
            head_prob = binomial(1,ht_per)
            
        entity_ID_list = list(self.entity_to_id.values())
        #corrupted_Index = np.asarray( random.sample(entity_ID_list, len(entity_ID_list)) )
        corrupted_Index = np.random.choice(len(entity_ID_list), batch_len)
        head_corrupt = np.where(head_prob >0.5)
        tail_corrupt = np.where(head_prob <=0.5)
        
        batch_negative_matrix[head_corrupt,0] = corrupted_Index[head_corrupt]
        
        batch_negative_matrix[tail_corrupt,2] = corrupted_Index[tail_corrupt]      
        
        for i in range(len(batch_negative_matrix)):            
            val = batch_negative_matrix[i,:]
            if (val[0], val[1], val[2]) in self.triplets_train_pool:
                while True:
                    if (val[0], val[1], val[2]) not in self.triplets_train_pool:
                        break
                    else:
                        if i in head_corrupt[0]:
                            val[0] = np.random.choice(entity_ID_list, 1)
                            batch_negative_matrix[i,0] = val[0]
                        else:
                            val[2] = np.random.choice(entity_ID_list, 1)
                            batch_negative_matrix[i,2] = val[2]
        
        
        return np.matrix.tolist(batch_positive_matrix), np.matrix.tolist(batch_negative_matrix)

    
    def next_batch_validate(self, batch_size):
        batch_validate = random.sample(self.triplets_validate, batch_size)

        return batch_validate

    # for each triplet, this function create a list of triplets for head and tail
    # the truth of the triplet is always on the top 
    # such format is used to evaluate the prediction
    def next_batch_eval(self, triplet_evaluate):
        # construct two batches for head and tail prediction
        batch_predict_head = []
        # replacing head
        id_heads_corrupted = set(self.id_to_entity.keys())
        #id_heads_corrupted.remove(triplet_evaluate[0])  # remove the golden head
        for i in range(self.num_entity):
            head = i
            if ((head, triplet_evaluate[1], triplet_evaluate[2])) in self.triplets_total_pool:
                id_heads_corrupted.remove(head)
        
        batch_predict_head.extend([(head, triplet_evaluate[1], triplet_evaluate[2]) for head in id_heads_corrupted])
        
        batch_predict_head.extend([triplet_evaluate])
        
        batch_predict_tail = []
        # replacing tail
        id_tails_corrupted = set(self.id_to_entity.keys())
        #id_tails_corrupted.remove(triplet_evaluate[2])  # remove the golden tail
        for i in range(self.num_entity):
            tail = i
            if ((triplet_evaluate[0], triplet_evaluate[1], tail)) in self.triplets_total_pool:
                id_tails_corrupted.remove(tail)
        batch_predict_tail.extend([(triplet_evaluate[0], triplet_evaluate[1], tail) for tail in id_tails_corrupted])
        batch_predict_tail.extend([triplet_evaluate])
        
        batch_predict_relation = []
        id_relation_corrupted = set(self.id_to_relation.keys())
        for i in range(self.num_relation):
            relation = i
            if ((triplet_evaluate[0], relation, triplet_evaluate[2])) in self.triplets_total_pool:
                id_relation_corrupted.remove(relation)
        
        batch_predict_relation.extend([(triplet_evaluate[0], relation, triplet_evaluate[2]) for relation in id_relation_corrupted])
        batch_predict_relation.extend([triplet_evaluate])
        
        return batch_predict_head, batch_predict_tail, batch_predict_relation

