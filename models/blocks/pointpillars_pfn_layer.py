"""PFNLayer of PointPillarsNet"""

from mindspore import nn
from mindspore import ops


class PFNLayer(nn.Cell):
    """PFN layer"""
    def __init__(self, in_channels, out_channels, use_norm, last_layer):
        super().__init__()

        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2

        self.units = out_channels
        self.use_norm = use_norm

        if use_norm:
            self.norm = nn.BatchNorm2d(self.units, eps=1e-3, momentum=0.99)
        else:
            self.norm = ops.Identity()
        self.linear = nn.Dense(in_channels, self.units, has_bias=not use_norm)

        self.transpose = ops.Transpose()
        self.tile = ops.Tile()
        self.concat = ops.Concat(axis=2)
        self.expand_dims = ops.ExpandDims()
        self.argmax_w_value = ops.ArgMaxWithValue(axis=-1, keep_dims=True)

    def construct(self, inputs):
        """forward graph"""
        x = self.linear(inputs)
        x = self.norm(x.transpose((0, 3, 1, 2)))  # [bs, V, P, 4]
        x = ops.ReLU()(x)
        x_max = self.argmax_w_value(x)[1].transpose((0, 2, 3, 1))  # [bs, V, P, 4]
        if self.last_vfe:
            return x_max
        x_repeat = self.tile(x_max, (1, 1, inputs.shape[1], 1))  # [bs, V, P, 4]
        x_concatenated = self.concat([x, x_repeat])
        return x_concatenated


