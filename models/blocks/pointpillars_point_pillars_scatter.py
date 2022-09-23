"""PointPillarsScatter of PointPillarsNet"""

from mindspore import nn
from mindspore import ops


class PointPillarsScatter(nn.Cell):
    """PointPillars scatter"""
    def __init__(self, output_shape, num_input_features):
        super().__init__()
        self.output_shape = output_shape
        self.ny = output_shape[2]
        self.nx = output_shape[3]
        self.n_channels = num_input_features
        self.scatter_nd = ops.ScatterNd()
        self.concat = ops.Concat(axis=1)
        self.expand_dims = ops.ExpandDims()
        self.transpose = ops.Transpose()

    def construct(self, voxel_features, coords):
        """forward graph"""
        # Batch_canvas will be the final output.
        batch_size = voxel_features.shape[0]
        # z coordinate is not used, z -> batch
        for i in range(batch_size):  # [bs, v, p, 64]
            coords[i, :, 0] = i
        shape = (batch_size, self.ny, self.nx, 2, self.n_channels)
        batch_canvas = self.scatter_nd(coords, voxel_features, shape)  # [bs, v, p, 2, 64]
        batch_canvas = batch_canvas[:, :, :, 1]
        batch_canvas = self.transpose(batch_canvas, (0, 3, 1, 2))
        return batch_canvas

