'''
Configuration file using yacs
'''
from yacs.config import CfgNode as CN

_C = CN()

_C.MNV2 = True

_C.SYSTEM = CN()
_C.SYSTEM.SEED = 42
_C.SYSTEM.NUM_WORKERS = 1


_C.DATA = CN()
_C.DATA.CLEAR = False
_C.MNV2 = True
_C.DATA.PROC_ONLY = False
_C.DATA.MAKE = False

_C.MODEL = CN()
_C.MODEL.MODELS_PATH = "./models/"
_C.MODEL.RESUME_TRAINING = False

_C.TRAIN = CN()
_C.TRAIN.VIZ_PATH = "./reports/"
_C.TRAIN.NUM_CLASSES = 1
_C.TRAIN.BATCH_SIZE = 2
_C.TRAIN.NUM_EPOCHS = 2000
_C.TRAIN.LR = 0.0001

def get_cfg_defaults():
    """
    Get a yacs CfgNode object with default values
    """
    # Return a clone so that the defaults will not be altered
    # It will be subsequently overwritten with local YAML.
    return _C.clone()
