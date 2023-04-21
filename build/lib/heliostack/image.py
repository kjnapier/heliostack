from astropy.time import Time
from astropy.wcs import WCS
from astropy.coordinates import Angle
from astropy import units as u

from scipy.ndimage import gaussian_filter

import numpy as np
import torch
import kornia

class Image:
    
    def __init__(self, image: np.ndarray, weight: np.ndarray, wcs: WCS, epoch: Time, psf=None, device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32):

        fwhm = 4
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        # pre-convolve the image and weight with the PSF
        image = gaussian_filter(image, sigma=sigma)
        weight = gaussian_filter(weight, sigma=sigma)
        
        self.image = torch.tensor(image, device=device, dtype=dtype)
        self.weight = torch.tensor(weight, device=device, dtype=dtype)
        self.wcs = wcs
        self.epoch = epoch
        self.psf = psf
        self.__calc_ra_dec()

    def __calc_ra_dec(self):
        Y_size, X_size = self.image.shape
        Ycenter = round(Y_size/2)
        Xcenter = round(X_size/2)
        ra, dec = self.wcs.wcs_pix2world(Xcenter, Ycenter, 0)
        self.ra = Angle(ra, u.deg).rad
        self.dec = Angle(dec, u.deg).rad