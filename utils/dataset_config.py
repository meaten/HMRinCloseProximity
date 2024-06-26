import os
from yacs.config import CfgNode as CN


def dataset_config(filename=None) -> CN:
    """
    Get dataset config file
    Returns:
      CfgNode: Dataset config as a yacs CfgNode object.
    """
    cfg = CN(new_allowed=True)
    config_file = filename
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg