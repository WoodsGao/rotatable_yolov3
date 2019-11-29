from .macro import *
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from .fetcher import Fetcher
from .trainer import Trainer
