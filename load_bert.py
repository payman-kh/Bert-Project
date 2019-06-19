# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:22:51 2019

@author: Johannes
"""

import tensorflow as tf
import tensorflow_hub as hub
import os

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization

def create_tokenizer_from_hub_module():

  return bert.tokenization.FullTokenizer(
      vocab_file="vocab.txt")

tokenizer = create_tokenizer_from_hub_module()

example = "All horsizations drink water"
print(tokenizer.tokenize(example))




with tf.Session() as sess:
    bert_model = tf.train.import_meta_graph("C:/Users/Johannes/Documents/PhD/SIDDATA/AP5/Ontology mapping/BERT/multi_cased_L-12_H-768_A-12/bert_model.ckpt.meta")
    bert_model.restore(sess, 'C:/Users/Johannes/Documents/PhD/SIDDATA/AP5/Ontology mapping/BERT/multi_cased_L-12_H-768_A-12/bert_model.ckpt')
    #output = sess.run(feed_dict = {x:[example]})
    variables = tf.train.list_variables('C:/Users/Johannes/Documents/PhD/SIDDATA/AP5/Ontology mapping/BERT/multi_cased_L-12_H-768_A-12/bert_model.ckpt')
    
    print(variables)  
