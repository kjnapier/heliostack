from astropy.time import Time
from astropy.wcs import WCS

from scipy.ndimage import gaussian_filter

import numpy as np
import torch
import kornia

class Image:
    
    def __init__(self, image: np.ndarray, weight: np.ndarray, wcs: WCS, epoch: Time, psf=None, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32):
        
        image = gaussian_filter(image, sigma=1.5)
        weight = gaussian_filter(weight, sigma=1.5)

        self.image = torch.tensor(image, device=device, dtype=dtype)
        self.weight = torch.tensor(weight, device=device, dtype=dtype)
        self.wcs = wcs
        self.epoch = epoch
        self.psf = psf

