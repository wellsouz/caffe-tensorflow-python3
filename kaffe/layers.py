import re
import numbers
from collections import namedtuple
from .caffe import caffepb

from .shapes import *

LAYER_DESCRIPTORS = {

    # Caffe Types
    'AbsVal': shape_identity,
    'Accuracy': shape_scalar,
    'ArgMax': shape_not_implemented,
    'BatchNorm': shape_identity,
    'BNLL': shape_not_implemented,
    'Concat': shape_concat,
    'ContrastiveLoss': shape_scalar,
    'Convolution': shape_convolution,
    'Deconvolution': shape_not_implemented,
    'Data': shape_data,
    'Dropout': shape_identity,
    'DummyData': shape_data,
    'EuclideanLoss': shape_scalar,
    'Eltwise': shape_identity,
    'Exp': shape_identity,
    'Flatten': shape_not_implemented,
    'HDF5Data': shape_data,
    'HDF5Output': shape_identity,
    'HingeLoss': shape_scalar,
    'Im2col': shape_not_implemented,
    'ImageData': shape_data,
    'InfogainLoss': shape_scalar,
    'InnerProduct': shape_inner_product,
    'Input': shape_data,
    'LRN': shape_identity,
    'MemoryData': shape_mem_data,
    'MultinomialLogisticLoss': shape_scalar,
    'MVN': shape_not_implemented,
    'Pooling': shape_pool,
    'Power': shape_identity,
    'ReLU': shape_identity,
    'Scale': shape_identity,
    'Sigmoid': shape_identity,
    'SigmoidCrossEntropyLoss': shape_scalar,
    'Silence': shape_not_implemented,
    'Softmax': shape_identity,
    'SoftmaxWithLoss': shape_scalar,
    'Split': shape_not_implemented,
    'Slice': shape_not_implemented,
    'TanH': shape_identity,
    'WindowData': shape_not_implemented,
    'Threshold': shape_identity,
}

V1_TO_NEW = {
    35: 'AbsVal',
    1: 'Accuracy',
    30: 'ArgMax',
    2: 'BNLL',
    3: 'Concat',
    37: 'ContrastiveLoss',
    4: 'Convolution',
    5: 'Data',
    39: 'Deconvolution',
    6: 'Dropout',
    32: 'DummyData',
    7: 'EuclideanLoss',
    25: 'Eltwise',
    38: 'Exp',
    8: 'Flatten',
    9: 'HDF5Data',
    10: 'HDF5Output',
    28: 'HingeLoss',
    11: 'Im2col',
    12: 'ImageData',
    13: 'InfogainLoss',
    14: 'InnerProduct',
    15: 'LRN',
    29: 'MemoryData',
    16: 'MultinomialLogisticLoss',
    34: 'MVN',
    17: 'Pooling',
    26: 'Power',
    18: 'ReLU',
    19: 'Sigmoid',
    27: 'SigmoidCrossEntropyLoss',
    36: 'Silence',
    20: 'Softmax',
    21: 'SoftmaxLoss',
    22: 'Split',
    33: 'Slice',
    23: 'TanH',
    24: 'WindowData',
    31: 'Threshold',
}

LAYER_TYPES = list(LAYER_DESCRIPTORS.keys())

LayerType = type('LayerType', (), {t: t for t in LAYER_TYPES})

class NodeKind(LayerType):

    @staticmethod
    def map_raw_kind(layer):
        kind = layer.type
        if isinstance(layer, caffepb.V1LayerParameter):
            kind = V1_TO_NEW[layer.type]
        if kind in LAYER_TYPES:
            return kind
        return None

    @staticmethod
    def compute_output_shape(node):
        try:
            val = LAYER_DESCRIPTORS[node.kind](node)
            return val
        except NotImplementedError:
            raise KaffeError('Output shape computation not implemented for type: %s' % node.kind)


class NodeDispatchError(KaffeError):

    pass


class NodeDispatch(object):

    @staticmethod
    def get_handler_name(node_kind):
        if len(node_kind) <= 4:
            # A catch-all for things like ReLU and tanh
            return node_kind.lower()
        # Convert from CamelCase to under_scored
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', node_kind)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

    def get_handler(self, node_kind, prefix):
        name = self.get_handler_name(node_kind)
        name = '_'.join((prefix, name))
        try:
            return getattr(self, name)
        except AttributeError:
            raise NodeDispatchError('No handler found for node kind: %s (expected: %s)' %
                                    (node_kind, name))


class LayerAdapter(object):

    def __init__(self, layer, kind):
        self.layer = layer
        self.kind = kind

    @property
    def parameters(self):
        name = NodeDispatch.get_handler_name(self.kind)
        name = '_'.join((name, 'param'))
        try:
            return getattr(self.layer, name)
        except AttributeError:
            raise NodeDispatchError('Caffe parameters not found for layer kind: %s' % (self.kind))

    @staticmethod
    def get_kernel_value(scalar, repeated, idx, default=None):
        if scalar:
            return scalar
        if repeated:
            if isinstance(repeated, numbers.Number):
                return repeated
            if len(repeated) == 1:
                # Same value applies to all spatial dimensions
                return int(repeated[0])
            assert idx < len(repeated)
            # Extract the value for the given spatial dimension
            return repeated[idx]
        if default is None:
            raise ValueError('Unable to determine kernel parameter!')
        return default

    @property
    def kernel_parameters(self):
        assert self.kind in (NodeKind.Convolution, NodeKind.Pooling)
        params = self.parameters
        k_h = self.get_kernel_value(params.kernel_h, params.kernel_size, 0)
        k_w = self.get_kernel_value(params.kernel_w, params.kernel_size, 1)
        s_h = self.get_kernel_value(params.stride_h, params.stride, 0, default=1)
        s_w = self.get_kernel_value(params.stride_w, params.stride, 1, default=1)
        p_h = self.get_kernel_value(params.pad_h, params.pad, 0, default=0)
        p_w = self.get_kernel_value(params.pad_h, params.pad, 1, default=0)
        return KernelParameters(k_h, k_w, s_h, s_w, p_h, p_w)


KernelParameters = namedtuple('KernelParameters', ['kernel_h', 'kernel_w', 'stride_h', 'stride_w',
                                                   'pad_h', 'pad_w'])
