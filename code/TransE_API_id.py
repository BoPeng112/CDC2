#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 11:33:02 2018

@author: Feipeng Zhao

TransE API

"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 17:13:18 2017

@author: Feipeng Zhao

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import tensorflow as tf
from KG_Data import KG_Data
import math
import numpy as np
import pdb
import gc

class TransE(object):
    def __init__(self, config):
        self.lr = lr = config.lr
        self.batch_size = batch_size = config.batch_size
        self.num_epoch = num_epoch = config.num_epoch
        self.margin = margin = config.margin
        self.embedding_dimension = embedding_dimension = config.embedding_dimension
        self.relation_dimension = embedding_dimension = config.embedding_dimension
        self.evaluate_size = evaluate_size = config.evaluate_size
        self.sampling_method = sampling_method = config.sampling_method
        self.metric = metric = config.metric
        self.log_path = log_path = config.log_path
        
        self.kg_data_path = config.kg_data_path
        
        self.load_data(self.kg_data_path)
        
        
        self.build_graph()
         
        # create a session
        self.sess = self.create_session()
        
        # save model
        self.model_path = model_path = config.model_path
        
    
    def __del__(self):
        self.sess.close()
        del self.sess
        print('session closed.')
    
    def load_data(self, data):
        self.dataset = KG_Data(data_dir=self.kg_data_path, negative_sampling= self.sampling_method)
        
        
    # create session function
    def create_session(self):
        
        gpu_options = tf.GPUOptions()
        gpu_options.allow_growth=True     
        sess = tf.Session(graph = self.kg_graph, config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
        return sess
    
    # given triplet return a score by using TransE
    def get_score(self, id_triplets):
        
        # get embedding from the graph
        #with tf.variable_scope('embedding', reuse=True):
        #    embedding_entity = tf.get_variable(name='entity')
        #    embedding_relation = tf.get_variable(name='relation')
            
        
        embedding_head = tf.nn.embedding_lookup(self.embedding_entity, id_triplets[:, 0])
        embedding_relation = tf.nn.embedding_lookup(self.embedding_relation, id_triplets[:, 1])
        embedding_tail = tf.nn.embedding_lookup(self.embedding_entity, id_triplets[:, 2])
        
        if self.metric == 'L2':
            dissimilarity = tf.sqrt(tf.reduce_sum(tf.square(embedding_head + embedding_relation
                                                            - embedding_tail), axis=1))
        else:  # default: L1
            dissimilarity = tf.reduce_sum(tf.abs(embedding_head + embedding_relation - embedding_tail), axis=1)

        score = -dissimilarity
        
        return score    
    
    # inference function that get positive score and negative score
    def inference(self, id_triplets_positive, id_triplets_negative):
        score_positive = self.get_score(id_triplets_positive)
        score_negative = self.get_score(id_triplets_negative)
        
        return score_positive, score_negative
    
    
    # calculate loss of batch positive triplets and negative triplets
    def get_loss(self, score_positive, score_negative):
        
        margin = tf.constant(
            self.margin,
            dtype=tf.float32,
            shape=None,
            name='margin'
        )
        
        loss = tf.reduce_mean(tf.nn.relu(margin + score_negative - score_positive), name='max_margin_loss')
        return loss 
    
    
    def evaluation(self, id_triplets_predict_head, id_triplets_predict_tail, id_triplets_predict_relation):
       # get one single triplet and do evaluation
        prediction_head = self.get_score(id_triplets_predict_head)
        prediction_tail = self.get_score(id_triplets_predict_tail)
        prediction_relation = self.get_score(id_triplets_predict_relation)
        
        return prediction_head, prediction_tail, prediction_relation
    
    
    # create graph for TransE
    def build_graph(self):
        
        #print('reset all variables...')
        #tf.reset_default_graph()
        print('build TransE graph...')
        
        self.kg_graph = tf.Graph()
        with self.kg_graph.as_default():
            # generate placeholders for inputs
            with tf.variable_scope('input'):
                self.id_triplets_positive = tf.placeholder(dtype = tf.int32, shape = [None,3], name = 'triplets_positive')
                self.id_triplets_negative = tf.placeholder(dtype = tf.int32, shape = [None,3], name = 'triplets_negative')
                
                self.id_triplets_predict_head = tf.placeholder(dtype = tf.int32, shape = [None,3], name = 'triplets_predict_head')
                self.id_triplets_predict_tail = tf.placeholder(dtype = tf.int32, shape = [None,3], name = 'triplets_predict_tail')
                self.id_triplets_predict_relation = tf.placeholder(dtype = tf.int32, shape = [None,3], name = 'triplets_predict_relation')
                
            # generate variables for embeddings
            bound = 6.0 / math.sqrt(self.embedding_dimension)
            with tf.variable_scope('embedding'):
                self.embedding_entity = tf.Variable(name = 'entity', initial_value = tf.random_uniform(shape = [self.dataset.num_entity, self.embedding_dimension], minval = -bound, maxval = bound) )
                self.embedding_relation = tf.Variable(name = 'relation', initial_value = tf.random_uniform(shape = [self.dataset.num_relation, self.embedding_dimension], minval = -bound, maxval = bound) )            
                
            
            with tf.name_scope('normalization'):
                self.normalize_entity_op = self.embedding_entity.assign(tf.nn.l2_normalize(self.embedding_entity, dim = 1))
                self.normalize_relation_op = self.embedding_relation.assign(tf.nn.l2_normalize(self.embedding_relation, dim = 1))
                
            
            with tf.name_scope('inference'):
                score_positive, score_negative = self.inference(self.id_triplets_positive, self.id_triplets_negative)
    
            
            with tf.name_scope('loss'):
                self.loss = self.get_loss(score_positive, score_negative)
                tf.summary.scalar(name = self.loss.op.name, tensor=self.loss)
            
            with tf.name_scope('training'):
                self.global_step = tf.Variable(initial_value=tf.constant(0, shape=[]), trainable=False, name = 'global_step')
                self.learning_rate = tf.Variable(self.lr, trainable=False, name='learning_rate')
                self.lr_schedule = tf.assign(self.learning_rate, self.lr)
                self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, self.global_step)
                self.init_op = tf.global_variables_initializer()
                self.merge_summary = tf.summary.merge_all()
            
            with tf.name_scope('testing'):
                self.predict_head, self.predict_tail, self.predict_relation = self.evaluation(self.id_triplets_predict_head, self.id_triplets_predict_tail, self.id_triplets_predict_relation)
        
            self.saver = tf.train.Saver() 
    # training model
    def train_model(self):
        
        # run the initial operation
        print('initializing all variables...')
        self.sess.run(self.init_op)
        
        print('all variables initialized')
        
        log_f =  open(self.log_path, 'w+')
        
        num_batch = self.dataset.num_triplets_train // self.batch_size
            
        #validate_data = dataset.triplets_validate[0:self.evaluate_size-1]
        #test_data = dataset.triplets_test
            
        # training
        print('start training...')
        epoch = 0
        start_total = time.time()
            
        init_learning_rate = self.lr
        for epoch in range(self.num_epoch):
            
            loss_epoch = 0
            start_train = time.time()
            
            if epoch > 500:
                lr_decay = 0.8
                self.lr = max(init_learning_rate * 0.1, self.lr * lr_decay)
                self.sess.run(self.lr_schedule)
            
            for batch in range(num_batch):
                
                self.sess.run(self.normalize_entity_op)
                self.sess.run(self.normalize_relation_op)
                
                batch_positive, batch_negative = self.dataset.next_batch_train_parallel(self.batch_size)
                
                feed_dict_train = {self.id_triplets_positive: batch_positive, self.id_triplets_negative: batch_negative}
                
                # run optimize op, loss op and summary op
                _, loss_batch, summary = self.sess.run([self.train_op, self.loss, self.merge_summary], feed_dict = feed_dict_train)
                
                loss_epoch += loss_batch
               
                
            end_train = time.time()
            
            if (epoch + 1) % 10 == 0:
                print('epoch {}, learning rate {:.5f}, mean triplet loss: {:.3f}, time elapsed last epoch: {:.3f}s'.format(epoch+1, self.lr, loss_epoch/num_batch, end_train-start_train))
                log_f.write('epoch {}, learning rate {:.5f}, mean triplet loss: {:.3f}, time elapsed last epoch: {:.3f}s \n'.format(epoch+1, self.lr, loss_epoch/num_batch, end_train-start_train))
        
            # evaluate the model every 10 epoches:
            if (epoch + 1) % 100 == 0:
                
                # in order to save time, only test 500 triplets in validation set
                hit10_head, hit1_head, rank_head, hit10_tail, hit1_tail, rank_tail, hit10_relation, hit1_relation, rank_relation = self.evaluate_triplets(self.dataset.triplets_validate[0:self.evaluate_size])
                log_f.write('testing accuracy is hit10_head: %.3f hit1_head: %.3f rank_head: %.3f hit10_tail: %.3f hit1_tail: %.3f rank_tail: %.3f hit10_relation: %.3f hit1_relation: %.3f rank_relation: %.3f\n'%(hit10_head, hit1_head, rank_head, hit10_tail, hit1_tail, rank_tail, hit10_relation, hit1_relation, rank_relation) )
        
        end_total = time.time()
        
        print('total time elapsed: {:.3f}s'.format(end_total - start_total))
        
        self.save_model()
        print('training finished, model saved.')
        
        return hit10_head, hit1_head, rank_head, hit10_tail, hit1_tail, rank_tail, hit10_relation, hit1_relation, rank_relation    
     
    # write embedding
    def get_embedding(self):
        
        self.sess.run(self.normalize_entity_op)
        self.sess.run(self.normalize_relation_op)
        
        return self.sess.run(self.embedding_entity), self.sess.run(self.embedding_relation)
    
    
    # save model
    def save_model(self):
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
                      
        self.saver.save(self.sess, self.model_path)
        
    # load model
    def load_model(self):
        print("loading model.")
        
        #ckpt = tf.train.get_checkpoint_state(self.model_path)
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.model_path))
            
        if ckpt and ckpt.model_checkpoint_path:
            print('success to read a model')
            
            self.saver.restore(self.sess, self.model_path)
            #print("success to read a model")
            return True
        else:
            print("failed to load a model")
            return False
        
    # testing model
    def test_model(self):
        hit10_head, hit1_head, rank_head_mean, mrr_head_mean, hit10_tail, hit1_tail, rank_tail_mean, mrr_tail_mean, hit10_relation, hit1_relation, rank_relation_mean, mrr_relation_mean = self.evaluate_triplets(self.dataset.triplets_test)
        
        print('Final testing accuracy is hit10_head: %.3f hit1_head: %.3f rank_head: %.3f mrr_head: %.3f hit10_tail: %.3f hit1_tail: %.3f rank_tail: %.3f mrr_tail: %.3f hit10_relation: %.3f hit1_relation: %.3f rank_relation: %.3f mrr_relation: %.3f\n'%(hit10_head, hit1_head, rank_head_mean, mrr_head_mean, hit10_tail, hit1_tail, rank_tail_mean, mrr_tail_mean, hit10_relation, hit1_relation, rank_relation_mean, mrr_relation_mean) )
        
        return hit10_head, hit1_head, rank_head_mean, mrr_head_mean, hit10_tail, hit1_tail, rank_tail_mean, mrr_tail_mean, hit10_relation, hit1_relation, rank_relation_mean, mrr_relation_mean
    
    # evaluate model
    def evaluate_triplets(self, evaluate_triplets):
        
        print('evaluating the current model...')
        
        f = open('test_shuffle', 'w')
        start_eval = time.time()
        rank_head = 0
        rank_tail = 0
        rank_relation = 0
        
        
        mrr_head = 0
        mrr_tail = 0
        mrr_relation = 0
        
        hit10_head = 0    
        hit10_tail = 0    
        hit10_relation = 0

        hit5_head = 0
        hit5_tail = 0
        hit5_relation = 0
    
        hit1_head = 0
        hit1_tail = 0
        hit1_relation = 0

        count_head = 0.0
        count_tail = 0.0
        count_relation = 0.0 

        f_head = open('head_prediction_DDI.tsv', 'w')
        f_tail = open('tail_prediction_DDI.tsv', 'w')
    
        test_size = len(evaluate_triplets)
        
        for triplet in evaluate_triplets:
            
            batch_predict_head, batch_predict_tail, batch_predict_relation = self.dataset.next_batch_eval(triplet)
            index_head = [i[0] for i in batch_predict_head].index(triplet[0])
            index_tail = [i[2] for i in batch_predict_tail].index(triplet[2])
            index_rel = [i[1] for i in batch_predict_relation].index(triplet[1])
            gc.collect()
            
            nhead = len(batch_predict_head)
            ntail = len(batch_predict_tail)
            nrel = len(batch_predict_relation)

            feed_dict_eval_1 = {self.id_triplets_predict_head: batch_predict_head[:nhead//2], self.id_triplets_predict_tail: batch_predict_tail[:ntail//2], self.id_triplets_predict_relation: batch_predict_relation[:nrel//2]}
            prediction_head_1, prediction_tail_1, prediction_relation_1 = self.sess.run([self.predict_head, self.predict_tail, self.predict_relation], feed_dict = feed_dict_eval_1)

            feed_dict_eval_2 = {self.id_triplets_predict_head: batch_predict_head[nhead//2:], self.id_triplets_predict_tail: batch_predict_tail[ntail//2:], self.id_triplets_predict_relation: batch_predict_relation[nrel//2:]}
            prediction_head_2, prediction_tail_2, prediction_relation_2 = self.sess.run([self.predict_head, self.predict_tail, self.predict_relation], feed_dict = feed_dict_eval_2)

            prediction_head = np.concatenate([prediction_head_1, prediction_head_2], axis = 0)
            prediction_tail = np.concatenate([prediction_tail_1, prediction_tail_2], axis = 0)
            prediction_relation = np.concatenate([prediction_relation_1, prediction_relation_2], axis = 0)
            
            # rank list of head and tail prediction
            count_head += prediction_head.flatten().tolist().count(1.0)
            #rank_head_current = (1.0-np.log(prediction_head)).flatten().argsort().argmax()
            #pdb.set_trace()
            rank_head_current = np.where((-prediction_head).flatten().argsort() == index_head)[0][0]
            rank_head = rank_head + rank_head_current
            mrr_head = mrr_head + 1.0/(rank_head_current+1.0)

            score_head = (-prediction_head).flatten()[-1]
            f_head.write("%s\t%s\t%s\t%.5f\t%d\n" % (triplet[0], triplet[1], triplet[2], score_head, rank_head_current))
            
            if rank_head_current < 10:
                hit10_head = hit10_head + 1

            if rank_head_current < 5:
                hit5_head = hit5_head + 1
        
            if rank_head_current < 1:
                hit1_head = hit1_head + 1
            #pdb.set_trace()
            #test_tail = (-prediction_tail_test).flatten().argsort().argmax()
            count_tail += prediction_tail.flatten().tolist().count(1.0)
            rank_tail_current = np.where((-prediction_tail).flatten().argsort() == index_tail)[0][0]
            rank_tail = rank_tail + rank_tail_current
            mrr_tail = mrr_tail + 1.0/(rank_tail_current+1.0)

            score_tail = (-prediction_tail).flatten()[-1]           
            f_tail.write("%s\t%s\t%s\t%.5f\t%d\n" % (triplet[0], triplet[1], triplet[2], score_tail, rank_tail_current))
            
            if rank_tail_current < 10:
                hit10_tail = hit10_tail + 1

            if rank_tail_current < 5:
                hit5_tail = hit5_tail + 1
        
            if rank_tail_current < 1:
                hit1_tail = hit1_tail + 1
            
            #test_rel = (1.0-np.log(prediction_relation_test)).flatten().argsort().argmax()
            count_relation += prediction_relation.flatten().tolist().count(1.0)
            rank_relation_current = np.where((-prediction_relation).flatten().argsort() == index_rel)[0][0]
            rank_relation = rank_relation + rank_relation_current
            mrr_relation = mrr_relation + 1.0/(rank_relation_current+1.0)
            
            if rank_relation_current < 10:
                hit10_relation = hit10_relation + 1

            if rank_relation_current < 5:
                hit5_relation = hit5_relation + 1

            if rank_relation_current < 1:
                hit1_relation = hit1_relation + 1

            f.write('%d\t%d\t%d\n' % (index_head, index_tail, index_rel))
            f.write('%d\t%d\t%d\n' % (rank_head_current, rank_tail_current, rank_relation_current))
        
        f_head.close()
        f_tail.close()
        rank_head_mean = rank_head / test_size
        mrr_head_mean = mrr_head / test_size
        hit10_head = hit10_head / test_size
        hit5_head = hit5_head / test_size
        hit1_head = hit1_head / test_size
    
        rank_tail_mean = rank_tail / test_size
        mrr_tail_mean = mrr_tail / test_size
        hit10_tail = hit10_tail / test_size
        hit5_tail = hit5_tail / test_size
        hit1_tail = hit1_tail / test_size
    
        rank_relation_mean = rank_relation / test_size
        mrr_relation_mean = mrr_relation / test_size
        hit10_relation = hit10_relation / test_size
        hit5_relation = hit5_relation / test_size
        hit1_relation = hit1_relation / test_size
        end_eval = time.time()
    
        print('head prediction mean rank: %.3f, mrr: %.3f, hit@10: %.3f, hit@5: %.3f, hit@1: %.3f '%(rank_head_mean, mrr_head_mean, hit10_head * 100, hit5_head * 100, hit1_head * 100))
        print('tail prediction mean rank: %.3f, mrr: %.3f, hit@10: %.3f, hit@5: %.3f, hit@1: %.3f '%(rank_tail_mean, mrr_tail_mean, hit10_tail * 100, hit5_tail * 100, hit1_tail * 100))
        #print('relation prediction mean rank: %.3f, mrr: %.3f, hit1@10: %.3f, hit@5: %.3f, hit@1: %.3f'%(rank_relation_mean, mrr_relation_mean, hit10_relation * 100, hit5_relation * 100, hit1_relation * 100))
        print('mean count head %.3f, mean count tail %.3f, mean count relation %.3f'%(count_head / test_size, count_tail / test_size, count_relation / test_size))
    
        print('time elapsed last evaluation: %.3f s'%(end_eval - start_eval))

        return hit10_head, hit1_head, rank_head_mean, mrr_head_mean, hit10_tail, hit1_tail, rank_tail_mean, mrr_tail_mean, hit10_relation, hit1_relation, rank_relation_mean, mrr_relation_mean
        
            
            
