
from models.cbamresnet import *
from models.pnasnet import *
from models.seresnext import *
from models.inceptionresnetv2 import *
from models.xception import *

__all__ = ['get_model']


_models = {
    'cbam_resnet18': cbam_resnet18,
    'cbam_resnet34': cbam_resnet34,
    'cbam_resnet50': cbam_resnet50,
    'cbam_resnet101': cbam_resnet101,
    'cbam_resnet152': cbam_resnet152,

    'pnasnet5large': pnasnet5large,

    'seresnext50_32x4d': seresnext50_32x4d,
    'seresnext101_32x4d': seresnext101_32x4d,
    'seresnext101_64x4d': seresnext101_64x4d,

    'xception': xception,
    'inceptionresnetv2': inceptionresnetv2,
}


def get_model(name, **kwargs):
    """
    Get supported model.

    Parameters:
    ----------
    name : str
        Name of model.

    Returns
    -------
    Module
        Resulted model.
    """
    name = name.lower()
    if name not in _models:
        raise ValueError('Unsupported model: {}'.format(name))
    net = _models[name](**kwargs)
    return net
