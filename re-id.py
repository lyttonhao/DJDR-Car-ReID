import sys
sys.path.insert(0, "mxnet/python/")
import mxnet as mx
import logging
import numpy as np
import argparse
import time
import random
from mxnet.optimizer import SGD
import loss_layers
from verifi_iterator import verifi_iterator
import importlib


def build_network(symbol, num_id, num_model):
    '''
    network structure
    '''
    # concat = internals["ch_concat_5b_chconcat_output"]
    pooling = mx.symbol.Pooling(
        data=symbol, kernel=(1, 1), global_pool=True,
        pool_type='avg', name='global_pool')
    flatten = mx.symbol.Flatten(data=pooling, name='flatten')

    fc1 = mx.symbol.FullyConnected(
        data=flatten, num_hidden=num_id, name='cls_fc1')
    softmax1 = mx.symbol.SoftmaxOutput(data=fc1, name='softmax1')
    fc2 = mx.symbol.FullyConnected(
        data=flatten, num_hidden=num_model, name='cls_fc2')
    softmax2 = mx.symbol.SoftmaxOutput(data=fc2, name='softmax2')

    l2 = mx.symbol.L2Normalization(data=flatten, name='l2_norm')

    verifi = mx.symbol.Custom(
        data=l2, grad_scale=1.0, threshd=args.verifi_threshd,
        op_type='verifiLoss', name='verifi')

    triplet = mx.symbol.Custom(
        data=l2, grad_scale=1.0, threshd=args.triplet_threshd,
        op_type='tripletLoss', name='triplet')

    outputs = [softmax1, softmax2]
    if args.verifi:
        outputs.append(verifi)
    if args.triplet:
        outputs.append(triplet)
    return mx.symbol.Group(outputs)


class Multi_Metric(mx.metric.EvalMetric):
    """Calculate accuracies of multi label"""
    def __init__(self, num=None, cls=1):
        super(Multi_Metric, self).__init__('multi-metric', num)
        self.cls = cls

    def update(self, labels, preds):
        # mx.metric.check_label_shapes(labels, preds)

        # if self.num != None:
        #     assert len(labels) == self.num
        for i in range(self.cls):
            pred_label = mx.nd.argmax_channel(
                preds[i]).asnumpy().astype('int32')
            label = labels[i].asnumpy().astype('int32')

            mx.metric.check_label_shapes(label, pred_label)

            if self.num is None:
                self.sum_metric += (pred_label.flat == label.flat).sum()
                self.num_inst += len(pred_label.flat)
            else:
                self.sum_metric[i] += (pred_label.flat == label.flat).sum()
                self.num_inst[i] += len(pred_label.flat)

        for i in range(self.cls, len(preds)):
            pred = preds[i].asnumpy()
            if self.num is None:
                self.sum_metric += np.sum(pred)
                self.num_inst += len(pred)
            else:
                self.sum_metric[i] += np.sum(pred)
                self.num_inst[i] += len(pred)


def get_imRecordIter(name, input_shape, batch_size, kv, shuffle=False, aug=False):
    dataiter = mx.io.ImageRecordIter(
        path_imglist="%s/%s.lst" % (args.data_dir, name),
        path_imgrec="%s/%s.rec" % (args.data_dir, name),
        mean_img="models/vid_mean.bin",
        rand_crop=aug,
        rand_mirror=aug,
        prefetch_buffer=4,
        preprocess_threads=3,
        shuffle=shuffle,
        label_width=2,
        data_shape=input_shape,
        batch_size=batch_size / 2,
        num_parts=kv.num_workers,
        part_index=kv.rank)

    return dataiter


def get_iterators(batch_size, input_shape, train, test, kv, gpus=1):
    '''
    use image list and rec file to generate data iterators
    '''
    train_dataiter1 = get_imRecordIter(
        '%s-even' % train, input_shape, batch_size,
        kv, shuffle=False, aug=True)
    train_dataiter2 = get_imRecordIter(
        '%s-rand' % train, input_shape, batch_size,
        kv, shuffle=True, aug=True)
    val_dataiter1 = get_imRecordIter(
        '%s-even' % test, input_shape, batch_size,
        kv, shuffle=False, aug=False)
    val_dataiter2 = get_imRecordIter(
        '%s-rand' % test, input_shape, batch_size,
        kv, shuffle=False, aug=False)

    return verifi_iterator(
        train_dataiter1, train_dataiter2, use_verifi=args.verifi, gpus=gpus), \
        verifi_iterator(
            val_dataiter1, val_dataiter2, use_verifi=args.verifi, gpus=gpus)

    # return train_dataiter2, val_dataiter2


def parse_args():
    parser = argparse.ArgumentParser(
        description='single domain car recog training')
    parser.add_argument('--gpus', type=str, default='0',
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--data-dir', type=str,
                        default="/mnt/hdd/lytton/mx_data/VehicleID",
                        help='data directory')
    parser.add_argument('--num-examples', type=int, default=119698,
                        help='the number of training examples')
    parser.add_argument('--num-id', type=int, default=13164,
                        help='the number of training examples')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='the batch size')
    parser.add_argument('--lr', type=float, default=.01,
                        help='the initial learning rate')
    parser.add_argument('--num-epoches', type=int, default=100,
                        help='the number of training epochs')
    parser.add_argument('--mode', type=str, default='tmp',
                        help='save names of model and log')
    parser.add_argument('--verifi-label', action='store_true', default=False,
                        help='if add verifi label')
    parser.add_argument('--verifi', action='store_true', default=False,
                        help='if use verifi loss')
    parser.add_argument('--triplet', action='store_true', default=False,
                        help='if use triplet loss')
    parser.add_argument('--verifi-threshd', type=float, default=0.9,
                        help='verification threshold')
    parser.add_argument('--triplet-threshd', type=float, default=0.9,
                        help='triplet threshold')
    parser.add_argument('--train-file', type=str, default="train-split1",
                        help='train file')
    parser.add_argument('--test-file', type=str, default="train-split2",
                        help='test file')
    parser.add_argument('--kv-store', type=str,
                        default='device', help='the kvstore type')
    parser.add_argument('--network', type=str,
                        default='inception-bn', help='network name')
    parser.add_argument('--model-load-epoch', type=int, default=126,
                        help='load the model on an epoch using the model-load-prefix')
    parser.add_argument('--model-load-prefix', type=str, default="inception-bn",
                        help='load model prefix')
    return parser.parse_args()


def load_checkpoint(prefix, epoch):
    # ssymbol = sym.load('%s-symbol.json' % prefix)
    save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return (arg_params, aux_params)


args = parse_args()

print args
batch_size = args.batch_size
num_epoch = args.num_epoches
devices = [mx.gpu(int(i)) for i in args.gpus.split(',')]
lr = args.lr
num_images = args.num_examples

arg_params, aux_params = load_checkpoint(
    'models/%s' % args.model_load_prefix, args.model_load_epoch)

symbol = importlib.import_module(
    'symbol_' + args.network).get_symbol()
net = build_network(symbol, num_id=args.num_id, num_model=251)
kv = mx.kvstore.create(args.kv_store)
train, val = get_iterators(
    batch_size=batch_size, input_shape=(3, 224, 224),
    train=args.train_file, test=args.test_file, kv=kv, gpus=len(devices))

stepPerEpoch = int(num_images * 2 / batch_size)
lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(
    step=[stepPerEpoch * x for x in [50, 75]], factor=0.1)
init = mx.initializer.Xavier(
    rnd_type='gaussian', factor_type='in', magnitude=2)

arg_names = net.list_arguments()
sgd = SGD(learning_rate=args.lr, momentum=0.9,
          wd=0.0001, clip_gradient=10, lr_scheduler=lr_scheduler,
          rescale_grad=1.0 / batch_size)

# args_lrscale = {}
# index = 0
# for name in arg_names:
#     if name != 'data' and name != 'softmax_label':
#         args_lrscale[index] = 1.0 if name.startswith('car_fc') else 1.0
#         index += 1

# sgd.set_lr_mult(args_lrscale)

logging.basicConfig(filename='log/%s.log' % args.mode, level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.info(args)

model = mx.model.FeedForward(
    symbol=net, ctx=devices, num_epoch=num_epoch, arg_params=arg_params,
    aux_params=aux_params, initializer=init, optimizer=sgd)

prefix = 'models/%s' % args.mode
num = 2
if args.verifi:
    num += 1
if args.triplet:
    num += 1
model.fit(X=train, eval_data=val, eval_metric=Multi_Metric(num=num, cls=2), logger=logger, epoch_end_callback=mx.callback.do_checkpoint(prefix),
          batch_end_callback=mx.callback.Speedometer(batch_size=batch_size))
