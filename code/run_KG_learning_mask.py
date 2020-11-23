#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from TransE_API_id import TransE
from CNN_API_mask import CNN



def run_KG_learning(data_path, data_name, model_name, output_path, config):
        
    kg_data_path = data_path + data_name
    with open(output_path + model_name + '_record.txt', 'w+') as f:
        f.write('embedding_dimension\t margin\t Hit10_Head\t Hit1_Head\t Rank_Head\t MRR_Head\t Hit10_Tail\t Hit1_Tail\t Rank_Tail\t  MRR_Tail\t Hit10_Relation\t Hit1_Relation\t Rank_Relation\t MRR_Relation\t \n')
        log_path = output_path + model_name +  '_log_dim_300batchsize_embedding_1000epochs_results_embedding_' + str(config.embedding_dimension) + '_margin_' + str(config.reg) + '_metric_' + config.metric + '.txt'
        config.log_path = log_path
        config.kg_data_path = kg_data_path
        
        # write model
        config.model_path = output_path + 'model/' + model_name +  '_model_dim_' + str(config.embedding_dimension) + '_margin_' + str(config.margin) + '_metric_' + config.metric + '.model'
        print('embedding dimension: %d, margin: %f\n'%(config.embedding_dimension, config.margin))
        
        if model_name == 'TransE':
            kg_m = TransE(config)
        elif model_name == 'TransH':
            kg_m = TransH(config)
        elif model_name == 'CNN':
            kg_m = CNN(config)
        
        #print('load model')
        #if not kg_m.load_model():  
        kg_m.train_model()
        
        hit10_head, hit1_head, rank_head, mrr_head, hit10_tail, hit1_tail, rank_tail, mrr_tail, hit10_relation, hit1_relation, rank_relation, mrr_relation = kg_m.test_model()
        
        embedding_entity, embedding_relation = kg_m.get_embedding()        
        entity_file = output_path + model_name +  '_log_dim_' + str(config.embedding_dimension) + '_margin_' + str(config.margin) + '_metric_' + config.metric + '.entity_embedding.txt'
        relation_file = output_path + model_name +  '_log_dim_' + str(config.embedding_dimension) + '_margin_' + str(config.margin) + '_metric_' + config.metric + 'relation_embedding.txt'
                            
        np.savetxt(entity_file,embedding_entity)
        np.savetxt(relation_file, embedding_relation)
                            
        print('testing accuracy is hit10_head: %f rank_head: %f mrr_head: %f hit10_tail: %f rank_tail: %f mrr_tail: %f \n'%(hit10_head, rank_head, mrr_head, hit10_tail, rank_tail, mrr_tail) )
        f.write('%d\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\n'%(config.embedding_dimension, config.margin, hit10_head, hit1_head, rank_head, mrr_head, hit10_tail, hit1_tail, rank_tail, mrr_tail, hit10_relation, hit1_relation, rank_relation, mrr_relation))

    f.close()
    #return (hit10_head + hit10_tail)/2    

class Config(object):
          
    #parameters setting
    # TransE and TransH part
    lr = 3e-3
    batch_size = 300
    num_epoch = 1000
    margin = 0.5
    embedding_dimension = 200      
    metric = 'L1'
    evaluate_size = 2000
    sampling_method = 'bern' 
    
    reg = 1e-3
    
##### main function to run the model
def main():
    
    reg_List = [0.0] 
    
    # make the configuration and run KG model
    for reg in reg_List:
        
        model_name ='CNN'
        print('reg%.4f'%reg)
        data_path = '../'
        data_name = 'dgi'
        output_path = '../' + data_name + '/' + model_name + '/'
    
        config = Config()
        config.reg = reg
    
        
        config.margin = 0.5
        config.sampling_method = 'bern'
        config.kernel_channel = 64
        config.fc_dim = 256
        config.train_keep_prob = 0.8
        config.test_keep_prob = 1.0
        config.embedding_dimension = 200
        
        run_KG_learning(data_path, data_name, model_name, output_path, config)
    
    
main()

