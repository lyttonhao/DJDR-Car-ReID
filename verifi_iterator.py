import mxnet as mx
import numpy as np


class verifi_iterator(mx.io.DataIter):
    '''
    Data iterator
    '''

    def __init__(self, data_iter1, data_iter2,
                 use_verifi=False, gpus=1):
        super(verifi_iterator, self).__init__()
        self.data_iter1 = data_iter1
        self.data_iter2 = data_iter2
        self.batch_size = self.data_iter1.batch_size * 2
        self.gpus = gpus
        self.use_verifi = use_verifi
        print "gpus", self.gpus

    @property
    def provide_data(self):
        provide_data = self.data_iter1.provide_data[0]
        shape = list(provide_data[1])
        shape[0] *= 2

        return [(provide_data[0], tuple(shape))]

    @property
    def provide_label(self):
        # provide_label = self.data_iter1.provide_label[0][1]
        # Different labels should be used here for actual application
        labels = [('softmax1_label', (self.batch_size,)), 
                  ('softmax2_label', (self.batch_size,))]
        if self.use_verifi:
            labels.append(('verifi_label', (self.batch_size,)))
        return labels

    def hard_reset(self):
        self.data_iter1.hard_reset()
        self.data_iter2.hard_reset()

    def reset(self):
        self.data_iter1.reset()
        self.data_iter2.reset()

    def next(self):
        batch1 = self.data_iter1.next()
        batch2 = self.data_iter2.next()

        def concat_array(data1, data2, gpus, ndarray=True):
            n = data1.shape[0]
            k = n / gpus
            data_lst = []
            for i in range(0, n, n / gpus):
                data_lst.append(data1[i:i + k])
                data_lst.append(data2[i:i + k])

            data = mx.nd.concatenate(data_lst, axis=0) if ndarray\
                else np.concatenate(data_lst, axis=0)

            return data

        data = concat_array(batch1.data[0], batch2.data[0], self.gpus)
        label = concat_array(batch1.label[0], batch2.label[0], self.gpus)
        label = label.T
        index = concat_array(
            batch1.index, -batch2.index, self.gpus, ndarray=False)

        labels = [label[0], label[1]]
        if self.use_verifi:
            labels.append(label[0])
        # print data.shape
        return mx.io.DataBatch(data=[data],
                               label=labels,
                               pad=batch1.pad + batch2.pad,
                               index=index)
