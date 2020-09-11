#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from flearn.servers.serveravg import FedAvg
from flearn.servers.serverfedl import FEDL
from flearn.trainmodel.models import *
from utils.plot_utils import *
import torch
torch.manual_seed(0)

