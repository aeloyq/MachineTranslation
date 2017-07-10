# -*- coding: utf-8 -*-
"""
Created on  三月 18 19:58 2017

@author: aeloyq
"""
from nnbuilder import *
import dictionary

source_vocab_size=30000
target_vocab_size=30000

source_emb_dim=620
target_emb_dim=620

enc_dim=1000
dec_dim=1000

sample.config.sample_freq=100
sample.config.sample_times=2
sample.config.sample_func=dictionary.mt_sample
sample.config.sample_from='test'
saveload.config.save_epoch=True
saveload.config.save_freq=10000
saveload.config.overwrite=False
saveload.config.load_file_name='Sat-Jul-01-15_39_54-2017.npz'
earlystop.config.valid_epoch=True
monitor.config.report_iter_frequence=1
monitor.config.report_iter=True
monitor.config.plot=True

config.vocab_source='./data/vocab.en-fr.en.pkl'
config.vocab_target='./data/vocab.en-fr.fr.pkl'
config.name='mt_bl'
config.data_path='./data/devsets_80.npy'
config.batch_size=80
config.valid_batch_size=64
config.max_epoches=1000
config.savelog=True
config.transpose_x=True
config.transpose_y=True
config.mask_x=True
config.mask_y=True
config.int_x=True
config.int_y=True

sgd.config.learning_rate=0.0001
adam.learning_rate=0.0001
adam.is_clip=True
adam.grad_clip_norm=1.

