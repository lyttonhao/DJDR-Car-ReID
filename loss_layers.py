import mxnet as mx
import numpy as np


class VerfiLoss(mx.operator.CustomOp):
    '''
    Verfication Loss Layer
    '''
    def __init__(self, grad_scale, threshd):
        self.grad_scale = grad_scale
        self.threshd = threshd
        self.eps = 1e-5

    def forward(self, is_train, req, in_data, out_data, aux):
        # print "forward"
        x = in_data[0]
        label = in_data[1]
        n = x.shape[0]
        ctx = x.context
        # y = out_data[0]
        # y[:] = 0
        # print y.shape
        y = np.zeros((x.shape[0], ))
        #y = mx.nd.array((n, ), ctx=ctx)
        for i in range(x.shape[0]):
            mask = np.zeros((n, ))
            mask[np.where(label == label[i])] = 1
            pos = np.sum(mask)
            mask = mx.nd.array(mask, ctx=ctx)
            diff = x[i] - x
            d = mx.nd.sqrt(mx.nd.sum(diff * diff, axis=1))
            d1 = mx.nd.maximum(0, self.threshd - d)
            z = mx.nd.sum(mask * d * d) / (pos + self.eps) \
                + mx.nd.sum((1 - mask) * d1 * d1) / (n - pos + self.eps)
            y[i] = z.asnumpy()[0]

        # y /= x.shape[0]
        self.assign(out_data[0], req[0], mx.nd.array(y, ctx=ctx))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # print "backward"
        x = in_data[0]
        label = in_data[1]
        n = x.shape[0]
        ctx = x.context
        grad = in_grad[0]
        grad[:] = 0
        for i in range(x.shape[0]):
            mask = np.zeros((1, n))
            mask[np.where(label == label[i])] = 1
            pos = np.sum(mask)
            mask = mx.nd.array(mask, ctx=ctx)
            diff = x[i] - x
            d = mx.nd.sqrt(mx.nd.sum(diff * diff, axis=1))
            g1 = mx.nd.minimum(0, (d - self.threshd) / (d + self.eps))

            z = mx.nd.dot((1 - mask) * g1.reshape([1, n]), diff)[0]
            # print grad[i].shape, z.shape
            # grad[i] = z
            # print "z"
            grad[i] = mx.nd.dot(mask, diff)[0] / (pos + self.eps)\
               + mx.nd.dot((1 - mask) * g1.reshape([1, n]), diff)[0] / (n - pos + self.eps)

        grad *= self.grad_scale


@mx.operator.register("verifiLoss")
class VerifiLossProp(mx.operator.CustomOpProp):
    def __init__(self, grad_scale=1.0, threshd=0.5):
        super(VerifiLossProp, self).__init__(need_top_grad=False)
        self.grad_scale = float(grad_scale)
        self.threshd = float(threshd)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0], )
        output_shape = (in_shape[0][0], )
        return [data_shape, label_shape], [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return VerfiLoss(self.grad_scale, self.threshd)


class TripletLoss(mx.operator.CustomOp):
    '''
    Triplet loss layer
    '''
    def __init__(self, grad_scale=1.0, threshd=0.5):
        self.grad_scale = grad_scale
        self.threshd = threshd

    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0]
        y = np.zeros((x.shape[0], ))
        ctx = x.context
        for i in range(x.shape[0] / 2):
            pid = i + 1 if i % 2 == 0 else i - 1
            nid = i + int(x.shape[0] / 2)
            pdiff = x[i] - x[pid]
            ndiff = x[i] - x[nid]
            y[i] = mx.nd.sum(pdiff * pdiff).asnumpy()[0] -\
                mx.nd.sum(ndiff * ndiff).asnumpy()[0] + self.threshd
            if y[i] < 0:
                y[i] = 0
        # y /= x.shape[0]
        self.assign(out_data[0], req[0], mx.nd.array(y, ctx=ctx))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        x = in_data[0]
        y = out_data[0]
        grad = in_grad[0]
        grad[:] = 0
        for i in range(x.shape[0] / 2):
            pid = i + 1 if i % 2 == 0 else i - 1
            nid = i + int(x.shape[0] / 2)

            if y[i] > 0:
                grad[i] += x[nid] - x[pid]
                grad[pid] += x[pid] - x[i]
                grad[nid] += x[i] - x[nid]

        grad *= self.grad_scale


@mx.operator.register("tripletLoss")
class TripletLossProp(mx.operator.CustomOpProp):
    def __init__(self, grad_scale=1.0, threshd=0.5):
        super(TripletLossProp, self).__init__(need_top_grad=False)
        self.grad_scale = float(grad_scale)
        self.threshd = float(threshd)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        # label_shape = (in_shape[0][0], )
        output_shape = (in_shape[0][0], )
        return [data_shape], [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return TripletLoss(self.grad_scale, self.threshd)