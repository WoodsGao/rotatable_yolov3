from .macro import *
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.autograd.set_detect_anomaly(True)
from .fetcher import Fetcher
from .trainer import Trainer
