"""Anchor generator builder"""
from src.core.anchor_generator import AnchorGeneratorStride


def build(anchor_config, a_type):
    """build anchor generator"""
    ag_type = a_type

    if ag_type == 'anchor_generator_stride':
        ag = AnchorGeneratorStride(
            sizes=list(anchor_config['sizes']),
            anchor_strides=list(anchor_config['strides']),
            anchor_offsets=list(anchor_config['offsets']),
            rotations=list(anchor_config['rotations']),
            match_threshold=anchor_config['matched_threshold'],
            unmatch_threshold=anchor_config['unmatched_threshold'],
            class_id=ag_type)
        return ag
    raise ValueError(" unknown anchor generator type")
