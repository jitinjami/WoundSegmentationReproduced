'''
Configuration file using yacs
'''
from yacs.config import CfgNode as CN

_C = CN()

_C.SEED = 42
_C.DATA_PATH = "./data/"
_C.MODELS_PATH = "./models/"
_C.VIZ_PATH = "./reports/"
_C.RESUME_TRAINING = True
_C.MAKE_DATA = False
_C.CLEAR_DATA = False
_C.NUM_CLASSES = 1
_C.BATCH_SIZE = 2
_C.NUM_WORKERS = 0
_C.NUM_EPOCHS = 1
_C.LR = 0.0001

def get_cfg_defaults():
    """
    Get a yacs CfgNode object with default values
    """
    # Return a clone so that the defaults will not be altered
    # It will be subsequently overwritten with local YAML.
    return _C.clone()
