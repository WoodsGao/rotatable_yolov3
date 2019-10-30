from .cv_utils import augments, config, utils
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'