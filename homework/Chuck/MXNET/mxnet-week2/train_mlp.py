import mxnet as mx
import numpy as np
import cv2
import logging
from data import get_mnist
from mlp_sym import get_mlp_sym
from mlp_sym import get_conv_sym

logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

TEST = False

