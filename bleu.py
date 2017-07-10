# -*- coding: utf-8 -*-
"""
Created on  三月 18 19:58 2017

@author: aeloyq
"""


from nnbuilder import *
import dictionary

import progressbar
bar = progressbar.ProgressBar()



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
config.batch_size=40

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


n=1
unk=False

model.build()
model.output.gen_sample(n)
datas=np.load(config.data_path)
nd=len(datas[2])
if  unk:
    new_list=[]
    for idx,ds,dt in zip(range(nd),datas[2], datas[5]):
        if not(1 in ds or 1 in dt):
            new_list.append(idx)
    newx=[]
    newy=[]
    for idx in new_list:
        newx=[datas[2][i] for i in new_list]
        newy=[datas[5][i] for i in new_list]
    datas = [datas[0],datas[1],newx,datas[3],datas[4],newy]


print 'compiling'
fs=theano.function(model.inputs,model.output.sample, on_unused_input='ignore',
                                updates=model.output.sample_updates)
print 'compile ok'



mbs=get_minibatches_idx(datas)
mbs=mbs[2]
savelist = [name for name in os.listdir('./{}/save/epoch'.format(config.name)) if name.endswith('.npz')]
def cp(x,y):
    xt=os.stat('./{}/save/epoch/'.format(config.name)+x)
    yt=os.stat('./{}/save/epoch/'.format(config.name)+y)
    if xt.st_mtime>yt.st_mtime :return 1
    else:return -1
savelist.sort(cp)


def bleutest(save):
    ss_list=[]
    s_text=''
    saveload.config.load_npz(model,save)
    print 'bleuing'

    bar = progressbar.ProgressBar()
    for idx,index in bar(mbs):
        data = prepare_data(datas[2], datas[5], index)
        ss = fs(*data)
        for s_ in ss:
            st=dictionary.mt_bleu(s_,unk)
            s_text+=st+'\r\n'




    print 'bleu ok'

    print 'dumping'
    f=open('./bleu/{}.txt'.format(str(n)),'wb')
    f.write(s_text+"\r\n")
    f.close()
    '''
    '''
    print 'dump ok'


    print'bleu testing'
    import os
    if os.path.exists('bleutmp'):os.remove('bleutmp')
    os.system('perl ./bleu/mb.perl ./bleu/t.txt < ./bleu/{}.txt >> ./bleutmp'.format(str(n)))
    f=open('bleutmp','rb')
    s=f.read()
    print s
    f.close()

t_text=''
for s in bar(datas[-1]):
    st=dictionary.mt_bleu(s)
    t_text+=st+'\r\n'
f=open('./bleu/t.txt','wb')
f.write(t_text+"\r\n")
f.close()
for sv in savelist:
    bleutest('epoch/'+sv.replace('.npz',''))