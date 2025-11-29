from 

https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-and-highlighting-code-blocks

I'm trying to use the Theano library in python to do some experiments with Deep Belief Networks. I use the code in this address: DBN full code. This code use the MNIST Handwritten database. This file is already in pickle format. It is unpicked in:

1. train_set
2. valid_set
3. test_set

Which is further unpickled in:

1. train_set_x, train_set_y = train_set
2. valid_set_x, valid_set_y = valid_set
3. test_set_x, test_set_y = test_set

Please can someone give me the code that constructs this dataset in order to create my own? The DBN example I use needs the data in this format and I don't know how to do it. if anyone has any ideas how to fix this, please tell me.

Here is my code:

python
```
from datetime import datetime
import time
import os
from pprint import pprint
import numpy as np
import gzip, cPickle
import theano.tensor as T
from theano import function


os.system("cls")

filename = "completeData.txt"


f = open(filename,"r")
X = []
Y = []

for line in f:
        line = line.strip('\n')  
        b = line.split(';')
        b[0] = float(b[0])
        b[1] = float(b[1])
        b[2] = float(b[2])
        b[3] = float(b[3])
        b[4] = float(b[4])
        b[5] = float(b[5])
        b[6] = float(b[6])
        b[7] = float(b[7])
        b[8] = float(b[8])
        b[9] = float(b[9])
        b[10] = float(b[10])
        b[11] = float(b[11])
        b[12] = float(b[12])
        b[13] = float(b[13])
        b[14] = float(b[14])
        b[15] = float(b[15])
        b[17] = int(b[17])
        X.append(b[:16])
        Y.append(b[17])

Len = len(X);
X = np.asmatrix(X)
Y = np.asarray(Y)

sizes = [0.8, 0.1, 0.1]
arr_index = int(sizes[0]*Len)
arr_index2_start = arr_index + 1
arr_index2_end = arr_index + int(sizes[1]*Len)
arr_index3_start = arr_index2_start + 1

"""
train_set_x = np.array(X[:arr_index])
train_set_y = np.array(Y[:arr_index])

val_set_x = np.array(X[arr_index2_start:arr_index2_end])
val_set_y = np.array(Y[arr_index2_start:arr_index2_end])

test_set_x = np.array(X[arr_index3_start:])
test_set_y = np.array(X[arr_index3_start:])

train_set = train_set_x, train_set_y
val_set = val_set_x, val_set_y
test_set = test_set_x, test_set_y
"""
x = T.dmatrix('x')
z = x
t_mat = function([x],z)

y = T.dvector('y')
k = y
t_vec = function([y],k)

train_set_x = t_mat(X[:arr_index].T)
train_set_y = t_vec(Y[:arr_index])
val_set_x = t_mat(X[arr_index2_start:arr_index2_end].T)
val_set_y = t_vec(Y[arr_index2_start:arr_index2_end])
test_set_x = t_mat(X[arr_index3_start:].T)
test_set_y = t_vec(Y[arr_index3_start:])

train_set = train_set_x, train_set_y
val_set = val_set_x, val_set_y
test_set = test_set_x, test_set_y

dataset = [train_set, val_set, test_set]

f = gzip.open('..\..\..\data\dex.pkl.gz','wb')
cPickle.dump(dataset, f, protocol=-1)
f.close()

pprint(train_set_x.shape)

print('Finished\n')
```

Answers:

A .pkl file is not necessary to adapt code from the Theano tutorial to your own data. You only need to mimic their data structure.

Quick fix
Look for the following lines. It's line 303 on DBN.py.

python
```
datasets = load_data(dataset)
train_set_x, train_set_y = datasets[0]
Replace with your own train_set_x and train_set_y.
```

python
```
my_x = []
my_y = []
with open('path_to_file', 'r') as f:
    for line in f:
        my_list = line.split(' ') # replace with your own separator instead
        my_x.append(my_list[1:-1]) # omitting identifier in [0] and target in [-1]
        my_y.append(my_list[-1])
train_set_x = theano.shared(numpy.array(my_x, dtype='float64'))
train_set_y = theano.shared(numpy.array(my_y, dtype='float64'))

```


Adapt this to your input data and the code you're using.

The same thing works for cA.py, dA.py and SdA.py but they only use train_set_x.

Look for places such as n_ins=28 * 28 where mnist image sizes are hardcoded. Replace 28 * 28 with your own number of columns.

Explanation
This is where you put your data in a format that Theano can work with.

python
```
train_set_x = theano.shared(numpy.array(my_x, dtype='float64'))
train_set_y = theano.shared(numpy.array(my_y, dtype='float64'))
```

shared() turns a numpy array into the Theano format designed for efficiency on GPUs.

dtype='float64' is expected in Theano arrays.

More details on basic tensor functionality.

.pkl file
The .pkl file is a way to save your data structure.

You can create your own.

python
```
import cPickle
f = file('my_data.pkl', 'wb')
    cPickle.dump((train_set_x, train_set_y), f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()
```