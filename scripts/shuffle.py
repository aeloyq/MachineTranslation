import numpy as np

batch = 80
maxlen = 50
name = 'devsets'
data = np.load('./' + name + '.npz')
sets = data['arr_0']
newsets=[]
print 'Loading data ...'
print 'Data loaded,ok'


def len_argsort(seq):
    return sorted(range(len(seq)), key=lambda x: len(seq[x]))


for index in range(3):

    print 'clipping to max length...'
    new_pair_set_x = []
    new_pair_set_y = []
    for x, y in zip(sets[index], sets[index + 3]):
        if len(x) < maxlen + 1 and len(y) < maxlen + 1:
            new_pair_set_x.append(x)
            new_pair_set_y.append(y)
    pair_set = new_pair_set_x
    pair_sety = new_pair_set_y
    del new_pair_set_x, new_pair_set_y
    print 'clipped,ok'

    n = len(pair_set)
    print 'sorting by length...'
    sorted_index = len_argsort(pair_set)
    pair_set = [pair_set[i] for i in sorted_index]
    pair_sety = [pair_sety[i] for i in sorted_index]
    blocks = (n - 1) // batch + 1
    idx = np.arange(blocks - 1)
    np.random.shuffle(idx)
    new_pair_set_x = []
    new_pair_set_y = []
    for i in idx:
        for j in range(batch):
            new_pair_set_x.append(pair_set[i * batch + j])
            new_pair_set_y.append(pair_sety[i * batch + j])
    for i in range(n % batch):
        new_pair_set_x.append(pair_set[(blocks - 1) * batch + i])
        new_pair_set_y.append(pair_sety[(blocks - 1) * batch + i])
    newsets.append(new_pair_set_x)
    newsets.append(new_pair_set_y)

train_x,train_y, valid_x,valid_y, test_x, test_y = newsets

try:
    print 'sorted,ok'
    print 'Train sentence pairs loaded in total: %s' % (len(train_x))
    print 'Valid sentence pairs loaded in total: %s' % (len(valid_x))
    print 'Test sentence pairs loaded in total: %s' % (len(test_x))
    print 'Max length in trainsets: %s' % max([len(i) for i in train_x])
    print 'Mean length in trainsets: %s' % np.mean([len(i) for i in train_x])
except:
    pass
np.save('./' + name + '_{}.npy'.format(batch), [train_x, valid_x, test_x, train_y, valid_y, test_y])
