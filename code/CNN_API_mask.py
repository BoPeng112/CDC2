#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import tensorflow as tf
from KG_Data import KG_Data
import math
import numpy as np
from TransE_API_id import TransE
import pdb

class CNN(TransE):
    def __init__(self, config):
        self.dropout = 0.8
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
        
        self.reg = reg = config.reg
        # CNN part 
        self.kernel_channel = kernel_channel = config.kernel_channel
        self.fc_dim = fc_dim = config.fc_dim
        self.train_keep_prob = train_keep_prob = config.train_keep_prob
        self.test_keep_prob = test_keep_prob = config.test_keep_prob
        
        self.kg_data_path = config.kg_data_path
        
        self.load_data(self.kg_data_path)
        
        
        self.build_graph()
          
        # create a session
        self.sess = self.create_session()
        
        # save model
        self.model_path = model_path = config.model_path
        
        
        
    def inference(self, id_triplets_positive, id_triplets_negative):
        score_positive, score_pos_masked = self.get_score(id_triplets_positive, self.train_keep_prob, phase = True)
        score_negative, score_neg_masked = self.get_score(id_triplets_negative, self.train_keep_prob, phase = True)
        #score_negative, score_neg_masked = self.get_score(id_triplets_negative, 1.0, phase = True)
        
        return score_positive, score_negative, score_pos_masked, score_neg_masked

        

    def get_score(self, id_triplets, keep_prob, phase):
        
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides = [1,1,3,1], padding = 'VALID')
   
        def conv1d(x, W):
            return tf.nn.conv1d(x, W, stride = 1, padding = 'VALID')
    
        def max_pool_kg(x):
            return tf.nn.max_pool(x, ksize = [1,1,2,1], strides = [1,1,2,1], padding = 'SAME')

        def residue_fc(x, w, b):
            return tf.nn.relu(tf.matmul(x, w) + b) + x

        def single_score(embeddings, W_conv, b_conv, W_fc1, W_fc2, b_fc1, b_fc2, phase, keep_prob, masked):
            x = conv2d(embeddings, W_conv) + b_conv
            x = tf.nn.dropout(x, keep_prob)
            x = tf.nn.relu(x)
            x_flat = tf.reshape(x, [-1, self.out_dim])
            x_flat = tf.matmul(x_flat, W_fc1) + b_fc1
            if masked:
                x= tf.nn.relu(x_flat)
            else:
                x_flat = tf.nn.dropout(x_flat, keep_prob)
                x = tf.nn.relu(x_flat)
            score = tf.nn.sigmoid(tf.matmul(x, W_fc2) + b_fc2)

            return score
        
        # get embedding from the graph
        with tf.variable_scope('embedding', reuse=True):
            embedding_entity = tf.get_variable(name ='entity')
            embedding_relation = tf.get_variable(name='relation')
            W_conv = tf.get_variable(name = 'W_conv')
            b_conv = tf.get_variable(name = 'b_conv')
            W_fc1 = tf.get_variable(name = 'W_fc1')
            b_fc1 = tf.get_variable(name = 'b_fc1')
            W_fc2 = tf.get_variable(name = 'W_fc2')
            b_fc2 = tf.get_variable(name = 'b_fc2')
            
        embedding_head = tf.nn.embedding_lookup(embedding_entity, id_triplets[:,0], max_norm = 1.0)
        embedding_relation = tf.nn.embedding_lookup(embedding_relation, id_triplets[:,1], max_norm = 1.0)
        embedding_tail = tf.nn.embedding_lookup(embedding_entity, id_triplets[:,2], max_norm =1.0)
 
        masked_head = tf.nn.dropout(embedding_head, self.dropout)
        masked_tail = tf.nn.dropout(embedding_tail, self.dropout)
        masked_relation = tf.nn.dropout(embedding_relation, self.dropout)
        
        embedding_triplet = tf.concat([embedding_head, embedding_relation, embedding_tail],1)
        embedding_triplet_masked = tf.concat([masked_head, masked_relation, masked_tail],1)
        
        embedding_cnn = tf.reshape(embedding_triplet, [-1, 3, self.embedding_dimension, 1])
        embedding_cnn_masked = tf.reshape(embedding_triplet_masked, [-1, 3, self.embedding_dimension, 1])
        
        score = single_score(embedding_cnn, W_conv, b_conv, W_fc1, W_fc2, b_fc1, b_fc2, phase, keep_prob, masked = False)
        masked_score = single_score(embedding_cnn_masked, W_conv, b_conv, W_fc1, W_fc2, b_fc1, b_fc2, phase, keep_prob, masked = True)
        
        return score, masked_score

     # calculate loss of batch positive triplets and negative triplets
    def get_loss(self, score_positive, score_negative, score_pos_masked, score_neg_masked):
        
        margin = tf.constant(
            self.margin,
            shape=None,
            name='margin'
        )
        
        
        class_loss_pos = -tf.reduce_mean(tf.log(score_positive + 1e-32), name = 'Entropy_pos')
        class_loss_neg = -tf.reduce_mean(tf.log(1.0 - score_negative + 1e-32), name = 'Entropy_neg')

        class_loss_pos_mask = -tf.reduce_mean(tf.log(score_pos_masked + 1e-32), name = 'Entropy_pos_mask')
        class_loss_neg_mask = -tf.reduce_mean(tf.log(1.0 - score_neg_masked + 1e-32), name = 'Entropy_neg_mask')
            
        class_loss = class_loss_pos + class_loss_neg
        class_loss_mask = class_loss_pos_mask + class_loss_neg_mask

        return class_loss + class_loss_mask
    
    def evaluation(self, id_triplets_predict_head, id_triplets_predict_tail, id_triplets_predict_relation):
       
        prediction_head, _ = self.get_score(id_triplets_predict_head, self.test_keep_prob, phase = False)
        prediction_tail, _ = self.get_score(id_triplets_predict_tail, self.test_keep_prob, phase = False)
        prediction_relation, _ = self.get_score(id_triplets_predict_relation, self.test_keep_prob, phase = False)
        
        return prediction_head, prediction_tail, prediction_relation
    
    
    def build_graph(self):
    
    
        self.layer_width = (int)(self.embedding_dimension - 3)/1 + 1
        self.sub_layer_width = (int)((self.layer_width-1)/3) + 1      
        self.out_dim = self.sub_layer_width * self.kernel_channel
        
        print('build CNN graph...')
        # construct training graph
        self.kg_graph = tf.Graph()
        with self.kg_graph.as_default():
        
            # generate placeholders for the graph
            with tf.variable_scope('input'):
                self.id_triplets_positive = tf.placeholder(dtype = tf.int32, shape = [None,3], name = 'triplets_positive')
                self.id_triplets_negative = tf.placeholder(dtype = tf.int32, shape = [None,3], name = 'triplets_negative_neighbor')
            
                self.id_triplets_predict_head = tf.placeholder(dtype = tf.int32, shape = [None,3], name = 'triplets_predict_head')
                self.id_triplets_predict_tail = tf.placeholder(dtype = tf.int32, shape = [None,3], name = 'triplets_predict_tail')
                self.id_triplets_predict_relation = tf.placeholder(dtype = tf.int32, shape = [None,3], name = 'triplets_predict_relation')
        
            #bound = 6 / math.sqrt(self.embedding_dimension)
            with tf.variable_scope('embedding'):
            

                self.embedding_entity = tf.get_variable(shape = [self.dataset.num_entity, self.embedding_dimension], initializer = tf.contrib.layers.xavier_initializer(), name = 'entity' )
                self.embedding_relation = tf.get_variable(shape = [self.dataset.num_entity, self.embedding_dimension],initializer =tf.contrib.layers.xavier_initializer(),name = 'relation')                 
                
                self.W_conv = tf.get_variable(shape = [3, 3, 1, self.kernel_channel], initializer = tf.contrib.layers.xavier_initializer(), regularizer = tf.contrib.layers.l2_regularizer(self.reg), name = 'W_conv')
                self.b_conv = tf.get_variable(shape = [self.kernel_channel], initializer = tf.contrib.layers.xavier_initializer(), regularizer = tf.contrib.layers.l2_regularizer(self.reg), name = 'b_conv')
                
                self.W_fc1 = tf.get_variable(shape = [self.out_dim, self.fc_dim], initializer = tf.contrib.layers.xavier_initializer(), regularizer = tf.contrib.layers.l2_regularizer(self.reg), name = 'W_fc1')
                self.b_fc1 = tf.get_variable(shape = [self.fc_dim], initializer = tf.contrib.layers.xavier_initializer(), regularizer = tf.contrib.layers.l2_regularizer(self.reg), name = 'b_fc1')
                
                self.W_fc2 = tf.get_variable(shape = [self.fc_dim, 1], initializer = tf.contrib.layers.xavier_initializer(), regularizer = tf.contrib.layers.l2_regularizer(self.reg), name = 'W_fc2')
                self.b_fc2 = tf.get_variable(shape = [1], initializer = tf.contrib.layers.xavier_initializer(), regularizer = tf.contrib.layers.l2_regularizer(self.reg), name = 'b_fc2')
                
            
            with tf.name_scope('inference'):
                score_positive, score_negative, score_pos_masked, score_neg_masked = self.inference(self.id_triplets_positive, self.id_triplets_negative)
        
            with tf.name_scope('loss'):
                self.reg_loss = tf.losses.get_regularization_loss()
                self.loss = self.get_loss(score_positive, score_negative, score_pos_masked, score_neg_masked) + self.reg_loss
                #self.loss = self.get_loss(score_positive, score_negative)
                tf.summary.scalar(name = self.loss.op.name, tensor=self.loss)
            
            
            with tf.name_scope('training'):
                self.global_step = tf.Variable(initial_value=tf.constant(0, shape=[]), trainable=False, name = 'global_step')
                self.learning_rate = tf.Variable(self.lr, trainable=False, name='learning_rate')
                self.lr_schedule = tf.assign(self.learning_rate, self.lr)
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                #self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
                self.train_op = self.optimizer.minimize(self.loss, self.global_step)
                self.merge_summary = tf.summary.merge_all()
                self.init_op = tf.global_variables_initializer()
            
            with tf.name_scope('testing'):
                self.predict_head, self.predict_tail, self.predict_relation = self.evaluation(self.id_triplets_predict_head, self.id_triplets_predict_tail, self.id_triplets_predict_relation)

            self.saver = tf.train.Saver()
            print('graph constructing finished')

    # training model
    def train_model(self):
        
        # run the initial operation
        print('initializing all variables...')
        self.sess.run(self.init_op)
        print('all variables initialized')
        
        log_f =  open(self.log_path, 'w+')
        
        num_batch = self.dataset.num_triplets_train // self.batch_size
            
        # training
        print('start training...')
        
        start_total = time.time()
            
        init_learning_rate = self.lr
        ##self.saver.restore(self.sess, tf.train.latest_checkpoint('./CNN_DGI/'))
        entitys = self.sess.run(self.embedding_entity)
        np.savetxt('entity_embeddings_DGI.txt', entitys, delimiter = ',', fmt = '%.5f')
        print ('finish loading model')
        
        for epoch in range(self.num_epoch):
            
            loss_epoch = 0
            start_train = time.time()
            
            lr_decay = 0.998
            self.lr = max(3e-4, self.lr * lr_decay)
            self.sess.run(self.lr_schedule)
            
            for batch in range(num_batch):
                st = time.time()
                batch_positive, batch_negative = self.dataset.next_batch_train_parallel(self.batch_size)
                
                feed_dict_train = {self.id_triplets_positive: batch_positive, self.id_triplets_negative: batch_negative}
                
                # run optimize op, loss op and summary op
                _, loss_batch, summary, reg_loss = self.sess.run([self.train_op, self.loss, self.merge_summary, self.reg_loss], feed_dict = feed_dict_train)
                
                loss_epoch += loss_batch
                ed = time.time()
                #print ('%.3f' % (ed-st))
                
            end_train = time.time()
            
            if (epoch + 1) % 10 == 0:
                print('epoch {}, learning rate {:.5f}, mean triplet loss: {:.3f}, time elapsed last epoch: {:.3f}s'.format(epoch+1, self.lr, loss_epoch/num_batch, end_train-start_train))
                #print(reg_loss)
                log_f.write('epoch {}, learning rate {:.5f}, mean triplet loss: {:.3f}, time elapsed last epoch: {:.3f}s \n'.format(epoch+1, self.lr, loss_epoch/num_batch, end_train-start_train))
        
            # evaluate the model every 10 epoches:
            if (epoch + 1) % 50 == 0 or epoch == self.num_epoch - 1:
                
                self.saver.save(self.sess, './CNN_DGI/best_model', global_step=epoch)
                # in order to save time, only test 500 triplets in validation set
                
                hit10_head, hit1_head, rank_head, mrr_head, hit10_tail, hit1_tail, rank_tail, mrr_tail, hit10_relation, hit1_relation, rank_relation, mrr_relation = self.evaluate_triplets(self.dataset.triplets_validate[0:self.evaluate_size])
                log_f.write('testing accuracy is hit10_head: %.3f hit1_head: %.3f rank_head: %.3f mrr_head: %.3f hit10_tail: %.3f hit1_tail: %.3f rank_tail: %.3f mrr_tail: %.3f hit10_relation: %.3f hit1_relation: %.3f rank_relation: %.3f mrr_relation: %.3f\n'%(hit10_head, hit1_head, rank_head, mrr_head, hit10_tail, hit1_tail, rank_tail, mrr_tail, hit10_relation, hit1_relation, rank_relation, mrr_relation) )
        
        end_total = time.time()
        
        print('total time elapsed: {:.3f}s'.format(end_total - start_total))
        
        self.save_model()
        print('training finished, model saved.')
        
        #return hit10_head, hit1_head, rank_head, mrr_head, hit10_tail, hit1_tail, rank_tail, mrr_tail, hit10_relation, hit1_relation, rank_relation, mrr_relation   
     
    # write embedding
    def get_embedding(self):
        
        return self.sess.run(self.embedding_entity), self.sess.run(self.embedding_relation)
    














    
        
        
        
        
