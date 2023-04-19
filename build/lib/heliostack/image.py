from astropy.time import Time
from astropy.wcs import WCS

import numpy as np
import torch

class Image:
    
    def __init__(self, image: np.ndarray, weight: np.ndarray, wcs: WCS, epoch: Time, psf=None, device: torch.device = torch.device('cpu')):
        self.image = torch.tensor(image, device=device, dtype=torch.half)
        self.weight = torch.tensor(weight, device=device, dtype=torch.half)
        self.wcs = wcs
        self.epoch = epoch
        self.psf = psf

