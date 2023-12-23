import numpy as np 
import tensorflow as tf
def seq_generator(x,y,r,delay,shuffle = True,batch_size = 3):
    seqsx = np.array(list(tf.keras.utils.timeseries_dataset_from_array(list(range(len(x))),None,delay,1,1,1,start_index = 0)))
    idx = np.arange(min(len(seqsx),len(y)))
    idx = idx[:len(idx)-(len(idx)%batch_size)]
    idx = idx.reshape((len(idx)//batch_size,batch_size))
    shape = idx.shape
    while True:
      if shuffle:
        idx = idx.flatten()
        np.random.shuffle(idx)
        idx = idx.reshape(shape)
      for id in idx:
        movie_input  = x[seqsx[id].squeeze()]
        running_input = r[seqsx[id].squeeze()]
        target = y[id]
        yield ([movie_input,running_input],target)