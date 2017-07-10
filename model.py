# -*- coding: utf-8 -*-
"""
Created on  三月 18 19:58 2017

@author: aeloyq
"""

from configuration import *

model=model(source_vocab_size,Int2dX,Int2dY)
model.sequential(Float2dMask,Float2dMask)
model.add(embedding(source_emb_dim))
model.add(encoder(enc_dim))
model.add(dropout(0.8))
model.add(decoder(dec_dim,target_emb_dim,target_vocab_size))
model.add(dropout(0.8))



