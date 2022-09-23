"""RPN of PointPillarsNet"""

import numpy as np
from mindspore import nn
from mindspore import ops
from mindspore.common import dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops.primitive import constexpr


@constexpr
def _create_on_value():
    """create on value"""
    return Tensor(1.0, mstype.float32)

@constexpr
def _log16():
    """log(16)"""
    return ops.Log()(Tensor(16.0, mstype.float32))


class RPN(nn.Cell):
    """RPN"""
    def __init__(
            self,
            use_norm=True,
            num_class=2,
            layer_nums=(3, 5, 5),
            layer_strides=(2, 2, 2),
            num_filters=(128, 128, 256),
            upsample_strides=(1, 2, 4),
            num_upsample_filters=(256, 256, 256),
            num_input_filters=128,
            num_anchor_per_loc=2,
            encode_background_as_zeros=True,
            use_direction_classifier=True,
            use_bev=False,
            box_code_size=7,
    ):
        super().__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self.use_direction_classifier = use_direction_classifier
        self.use_bev = use_bev
        self._use_norm = use_norm

        if len(layer_nums) != 3:
            raise ValueError(f'Layer nums must be 3, got {layer_nums}')
        if len(layer_nums) != len(layer_strides):
            raise ValueError(f'Layer nums and layer strides must have same length, '
                             f'got {len(layer_nums)}, {len(layer_strides)} rescpectively')
        if len(layer_nums) != len(num_filters):
            raise ValueError(f'Layer nums and num filters must have same length, '
                             f'got {len(layer_nums)}, {len(num_filters)} respectively')
        if len(layer_nums) != len(upsample_strides):
            raise ValueError(f'Layer nums and upsample strides must have same length, '
                             f'got {len(layer_nums)}, {len(upsample_strides)} respectively')
        if len(layer_nums) != len(num_upsample_filters):
            raise ValueError(f'Layer nums and num upsample strides must have same length, '
                             f'got {len(layer_nums)}, {len(num_upsample_filters)} respectively')

        factors = []
        for i in range(len(layer_nums)):
            factors.append(np.prod(layer_strides[: i + 1]) // upsample_strides[i])

        if use_norm:
            batch_norm2d_class = nn.BatchNorm2d
        else:
            batch_norm2d_class = ops.Identity()

        block2_input_filters = num_filters[0]

        if use_bev:
            self.bev_extractor = nn.SequentialCell(
                nn.Conv2d(6, 32, 3, padding=1, pad_mode='pad', has_bias=not use_norm),
                batch_norm2d_class(32, eps=1e-3, momentum=0.99),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1, pad_mode='pad', has_bias=not use_norm),
                batch_norm2d_class(64, eps=1e-3, momentum=0.99),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )
            block2_input_filters += 64

        self.block1 = nn.SequentialCell(
            nn.Conv2d(
                num_input_filters,
                num_filters[0],
                3,
                padding=1,
                pad_mode='pad',
                stride=layer_strides[0],
                has_bias=not use_norm
            ),
            batch_norm2d_class(num_filters[0], eps=1e-3, momentum=0.99),
            nn.ReLU(),
        )
        for i in range(layer_nums[0]):
            self.block1.append(
                nn.Conv2d(num_filters[0], num_filters[0], 3, padding=1, pad_mode='pad', has_bias=not use_norm)
            )
            self.block1.append(batch_norm2d_class(num_filters[0]))
            self.block1.append(nn.ReLU())
        self.deconv1 = nn.SequentialCell(
            nn.Conv2dTranspose(
                num_filters[0],
                num_upsample_filters[0],
                upsample_strides[0],
                stride=upsample_strides[0],
                has_bias=not use_norm
            ),
            batch_norm2d_class(num_upsample_filters[0], eps=1e-3, momentum=0.99),
            nn.ReLU(),
        )
        self.block2 = nn.SequentialCell(
            nn.Conv2d(
                block2_input_filters,
                num_filters[1],
                3,
                padding=1,
                pad_mode='pad',
                stride=layer_strides[1],
                has_bias=not use_norm
            ),
            batch_norm2d_class(num_filters[1], eps=1e-3, momentum=0.99),
            nn.ReLU(),
        )
        for i in range(layer_nums[1]):
            self.block2.append(
                nn.Conv2d(num_filters[1], num_filters[1], 3, padding=1, pad_mode='pad', has_bias=not use_norm)
            )
            self.block2.append(batch_norm2d_class(num_filters[1], eps=1e-3, momentum=0.99))
            self.block2.append(nn.ReLU())
        self.deconv2 = nn.SequentialCell(
            nn.Conv2dTranspose(
                num_filters[1],
                num_upsample_filters[1],
                upsample_strides[1],
                stride=upsample_strides[1],
                has_bias=not use_norm
            ),
            batch_norm2d_class(num_upsample_filters[1], eps=1e-3, momentum=0.99),
            nn.ReLU(),
        )
        self.block3 = nn.SequentialCell(
            nn.Conv2d(
                num_filters[1],
                num_filters[2],
                3,
                padding=1,
                pad_mode='pad',
                stride=layer_strides[2],
                has_bias=not use_norm
            ),
            batch_norm2d_class(num_filters[2], eps=1e-3, momentum=0.99),
            nn.ReLU(),
        )
        for i in range(layer_nums[2]):
            self.block3.append(
                nn.Conv2d(num_filters[2], num_filters[2], 3, padding=1, pad_mode='pad', has_bias=not use_norm)
            )
            self.block3.append(batch_norm2d_class(num_filters[2], eps=1e-3, momentum=0.99))
            self.block3.append(nn.ReLU())
        self.deconv3 = nn.SequentialCell(
            nn.Conv2dTranspose(
                num_filters[2],
                num_upsample_filters[2],
                upsample_strides[2],
                stride=upsample_strides[2],
                has_bias=not use_norm
            ),
            batch_norm2d_class(num_upsample_filters[2], eps=1e-3, momentum=0.99),
            nn.ReLU(),
        )
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        self.conv_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, 1)
        self.conv_box = nn.Conv2d(sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(sum(num_upsample_filters), num_anchor_per_loc * 2, 1)

        self.transpose = ops.Transpose()
        self.concat = ops.Concat(axis=1)

    def construct(self, x, bev=None):
        """forward graph"""
        x = self.block1(x)
        up1 = self.deconv1(x)
        if self.use_bev:
            bev[:, -1] = ops.Log()(1 + bev[:, -1]) / _log16()
            bev[:, -1] = ops.clip_by_value(
                bev[:, -1],
                clip_value_min=bev[:, -1].min(),
                clip_value_max=_create_on_value()
            )
            x = self.concat([x, self.bev_extractor(bev)])
        x = self.block2(x)
        up2 = self.deconv2(x)
        x = self.block3(x)
        up3 = self.deconv3(x)
        x = self.concat([up1, up2, up3])
        # 以上的2Dbackbone已经设计结束，后面是head
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        box_preds = self.transpose(box_preds, (0, 2, 3, 1))
        cls_preds = self.transpose(cls_preds, (0, 2, 3, 1))

        if self.use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = self.transpose(dir_cls_preds, (0, 2, 3, 1))
            return box_preds, cls_preds, dir_cls_preds
        return box_preds, cls_preds
