"""builder of backbone, head, neck, etc..."""

from mindvision.engine.class_factory import ClassFactory, ModuleType


def build_backbone(cfg):
    """build backbone"""
    return ClassFactory.get_instance_from_cfg(cfg, ModuleType.BACKBONE)


def build_head(cfg):
    """build head"""
    return ClassFactory.get_instance_from_cfg(cfg, ModuleType.HEAD)


def build_neck(cfg):
    """build neck"""
    return ClassFactory.get_instance_from_cfg(cfg, ModuleType.NECK)


def build_classifier(cfg):
    """build classifier"""
    return ClassFactory.get_instance_from_cfg(cfg, ModuleType.CLASSIFIER)


def build_recognizer(cfg):
    """build recognizer"""
    return ClassFactory.get_instance_from_cfg(cfg, ModuleType.RECOGNIZER)
