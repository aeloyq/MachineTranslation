# -*- coding: utf-8 -*-
"""
Created on  三月 18 19:58 2017

@author: aeloyq
"""

from configuration import *

data=np.load(config.data_path)
result=train(datastream=data,model=model,algrithm=adam,extension=[monitor])