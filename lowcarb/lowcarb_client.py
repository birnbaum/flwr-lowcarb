from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

from lowcarb.carbon_sdk_webapi import get_forecast_batch

import requests

DEVICE = torch.device("cpu")


class Lowcarb_Client(object):
    def __init__(self, *args, **kwargs):
        super(Lowcarb_Client, self).__init__(*args, **kwargs)
        self.location = None

    def get_properties(self, config):
        return {'location': self.location}


def lowcarb(func):
    def wrapper(*args):
        properties = func(*args)
        properties['location'] = args[0].location

        return properties

    return wrapper
