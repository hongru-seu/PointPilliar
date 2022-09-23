"""voxel generator builder"""
from src.core.voxel_generator import VoxelGenerator


def build(voxel_cfg):
    """build voxel generator"""
    voxel_generator = VoxelGenerator(
        voxel_size=list(voxel_cfg['voxel_size']),
        point_cloud_range=list(voxel_cfg['point_cloud_range']),
        max_num_points=voxel_cfg['max_number_of_points_per_voxel']
    )
    return voxel_generator
