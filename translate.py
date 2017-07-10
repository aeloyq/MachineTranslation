# -*- coding: utf-8 -*-
"""
Created on  三月 18 19:58 2017

@author: aeloyq
"""

import nnbuilder
from nnbuilder.data import *
from nnbuilder.layers.simple import *
from nnbuilder.layers.sequential import *
from nnbuilder.algrithms import *
from nnbuilder.extensions import *
from nnbuilder.model import *
from nnbuilder.main import *
import dictionary
import tarfile
import os
import subprocess
import numpy as np
import cPickle as cp

config.name='mt'
config.data_path='./data/devsets_40.npy'
config.transpose_x=True
config.transpose_y=True
config.mask_x=True
config.mask_y=True
config.int_x=True
config.int_y=True
config.vocab_source='./data/vocab.en-fr.en.pkl'
config.vocab_target='./data/vocab.en-fr.fr.pkl'

source_vocab_size=30000
target_vocab_size=30000
source_emb_dim=620
target_emb_dim=620
enc_dim=1000
dec_dim=1000

model=model(source_vocab_size,Int2dX,Int2dY)
model.sequential(Float2dMask,Float2dMask)
model.add(embedding(source_emb_dim))
model.add(encoder(enc_dim))
model.add(dropout(0.8))
model.add(decoder(dec_dim,target_emb_dim,target_vocab_size))
model.add(dropout(0.8))


model.build()


mo=model.output
f=theano.function(model.inputs,mo.predict, on_unused_input='ignore',
                                updates=model.output.raw_updates)
print 'ok1'
#                                  0          1         2               3       4        5          6         7        8        9          10
model.output.gen_sample(1)
ft=theano.function(model.inputs,[mo.sample], on_unused_input='ignore',
                                updates=model.output.sample_updates)
print 'ok2'

model.output.gen_sample(2)
ftt=theano.function(model.inputs,[mo.sample, mo.y_mm, mo.y_mm_shifted, mo.prob,mo.score,mo.samples,mo.choice,mo.y_idx,mo.y_out,mo.y_idx_c,mo.y_pred,mo.score_sum], on_unused_input='ignore',
                                updates=model.output.sample_updates)

print 'ok3'
model.output.gen_sample(3)
fttt=theano.function(model.inputs,[mo.sample, mo.y_mm, mo.y_mm_shifted, mo.prob,mo.score,mo.samples,mo.choice,mo.y_idx,mo.y_out,mo.y_idx_c,mo.y_pred,mo.score_sum], on_unused_input='ignore',
                                updates=model.output.sample_updates)

print 'ok4'
model.output.gen_sample(12)
ftttt=theano.function(model.inputs,[mo.sample, mo.y_mm, mo.y_mm_shifted, mo.prob,mo.score,mo.samples,mo.choice,mo.y_idx,mo.y_out,mo.y_idx_c,mo.y_pred,mo.score_sum], on_unused_input='ignore',
                                updates=model.output.sample_updates)

print 'ok5'
saveload.config.load_npz(model)
data=np.load(config.data_path)
def test(n):
    d=prepare_data(data[0],data[3],[n])
    return f(*d),ft(*d),ftt(*d),fttt(*d),ftttt(*d)

'''
sample_function=theano.function(model.inputs,model.output.sample, on_unused_input='ignore',
                                updates=model.output.sample_updates)

import os
def sampling(sentence):
    s=prepare(sentence)

    ss=[dictionary.mt_s(s)]

    sss=prepare_data(ss, [[0]], [0])

    t=sample_function(*sss)

    st = dictionary.mt_bleu(t[0],True)

    print '\r\ngot result:\r\n'
    print '☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆'
    print st
    print '☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆'

def prepare(sentence):
    path='samplingtmp'
    tokenizer_file='./scripts/tokenizer.perl'
    if not os.path.exists(path): os.mkdir(path)
    f = open(path+'/source.en'.format(str(n)), 'wb')
    f.write(sentence)
    f.close()

    def tokenize_text_files(files_to_tokenize, tokenizer, OUTPUT_DIR):
        name = files_to_tokenize
        print ("Tokenizing file [{}]".format(name))
        out_file = os.path.join(
            OUTPUT_DIR, os.path.basename(name) + '.tok')
        print("...writing tokenized file [{}]".format(out_file))
        var = ["perl", tokenizer, "-l", name.split('.')[-1]]

        if os.path.exists(out_file):
            os.remove(out_file)

        with open(name, 'r') as inp:
            with open(out_file, 'w', 0) as out:
                subprocess.check_call(
                    var, stdin=inp, stdout=out, shell=False)
                print("wrote tokenized file [{}]".format(out_file))

    tokenize_text_files(path + '/source.en', tokenizer_file, path)
    f = open(path+'/source.en.tok'.format(str(n)), 'rb')
    s=f.readline()
    f.close()
    return s

while(True):
    a=raw_input('\r\n\r\nPlease enter the sentence:\r\n\r\n')
    sampling(a)
'''