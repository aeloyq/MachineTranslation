# -*- coding: utf-8 -*-
"""
Created on  四月 04 22:59 2017

@author: aeloyq
"""
import numpy as np
from itertools import izip
from nnbuilder import config


def invert_dict3(d):
    return dict(izip(d.itervalues(), d.iterkeys()))

s='en'
t='fr'
ds=np.load('./data/vocab.{}-{}.{}.pkl'.format(s,t,s))
dt=np.load('./data/vocab.{}-{}.{}.pkl'.format(s,t,t))
dsv=invert_dict3(ds)
dst=invert_dict3(dt)

def mt_sample(inputs,result,y):
    sample_str = "Sample:\r\n"
    sst=''
    tst=''
    mst=''
    for i in inputs:
        sst=sst+dsv[i[0]]+' '
    for i in y:
        tst=tst+dst[i[0]]+' '
    for i in result:
        mst=mst+dst[i[0]]+' '
    mst =mst.replace('&amp; apos ;','\'')
    mst =mst.replace('&amp; quot ;', '\"')
    sample_str += 'Source Sentence: {}\r\n'.format(sst)
    sample_str += 'Sample Sentence: {}'.format(mst)
    return sample_str,'   '+tst

def mt_bleu(result):

    ds = np.load(config.vocab_source)
    dt = np.load(config.vocab_target)
    dsv = invert_dict3(ds)
    dst = invert_dict3(dt)
    mst=''
    for i in result:
        if i!=0:
            mst=mst+dst[i]+' '
        else :
            break
    mst =mst.replace('&amp; apos ;','\'')
    mst =mst.replace('&amp; quot ;', '\"')
    return mst

def mt_s(result):
    ds = np.load(config.vocab_source)
    dt = np.load(config.vocab_target)
    mst=[]
    result.replace('\r','')
    result.replace('\n', '')
    for i in result.split(' '):
        try:
            mst.append(ds[i])
        except:
            mst.append(1)
    mst.append(0)
    return mst

def mt_t(result):
    ds = np.load(config.vocab_source)
    dt = np.load(config.vocab_target)
    dsv = invert_dict3(ds)
    dst = invert_dict3(dt)
    mst=''
    for i in result:
        if i!=0:
            mst=mst+dst[i]+' '
        else :
            break
    return mst