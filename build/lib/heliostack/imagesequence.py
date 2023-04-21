from .image import Image

import torch
from astropy import units as u
from astropy.wcs import WCS

class ImageSequence:

    def __init__(self, images: list[Image]):
        self.images = images
        
    @property
    def epochs(self):
        return torch.tensor([image.epoch.jd for image in self.images])
    
    @property
    def device(self):
        return self.images[0].image.device
    
    @property
    def dtype(self):
        return self.images[0].image.dtype
    
    def stack(self, ra_rate: float, dec_rate: float, reference_epoch: int = 0, EDGE_CUT_PIXELS: int = 250, pixel_scale: float = 0.27):

        ra_rate = (ra_rate * u.arcsec / u.hour).to(u.rad / u.day).value
        dec_rate = (dec_rate * u.arcsec / u.hour).to(u.rad / u.day).value
        pixel_scale = pixel_scale * u.arcsec.to(u.rad)

        t0 = self.epochs[reference_epoch]
        dts = self.epochs - t0
        ra0 = self.images[reference_epoch].ra
        dec0 = self.images[reference_epoch].dec
        ref_y_size, ref_x_size = self.images[reference_epoch].image.shape
        SS_wcs = self.images[reference_epoch].wcs.deepcopy()

        ras = ra0 + ra_rate * dts
        decs = dec0 + dec_rate * dts

        xs, ys = SS_wcs.wcs_world2pix(ras * u.rad.to(u.deg), decs * u.rad.to(u.deg), 0)

        Nx = int(max(xs) - min(xs)) + ref_x_size + 10 # add 10 pixels to be safe
        Ny = int(max(ys) - min(ys)) + ref_y_size + 10 # add 10 pixels to be safe

        y_ref = round((Ny - ref_y_size)/2)
        x_ref = round((Nx - ref_x_size)/2)
        SS_wcs.wcs.crpix[0] += x_ref
        SS_wcs.wcs.crpix[1] += y_ref

        
        psi = torch.zeros(Ny, Nx, device=self.device, dtype=self.dtype)
        phi = torch.zeros(Ny, Nx, device=self.device, dtype=self.dtype)

        for image, x, y in zip(self.images, xs, ys):
                            
            Y_size, X_size = image.image.shape

            bottom = round(Ny/2 - y) + EDGE_CUT_PIXELS
            top = round(Ny/2 - y + Y_size) - EDGE_CUT_PIXELS
            left = round(Nx/2 - x) + EDGE_CUT_PIXELS
            right = round(Nx/2 - x + X_size) - EDGE_CUT_PIXELS

            psi[bottom:top, left:right] += image.image[EDGE_CUT_PIXELS:-EDGE_CUT_PIXELS, EDGE_CUT_PIXELS:-EDGE_CUT_PIXELS]
            phi[bottom:top, left:right] += image.weight[EDGE_CUT_PIXELS:-EDGE_CUT_PIXELS, EDGE_CUT_PIXELS:-EDGE_CUT_PIXELS]
        
        s = psi / torch.sqrt(phi)

        # mask nans and infs
        s = torch.where(torch.isnan(s), torch.zeros_like(s), s)
        s = torch.where(torch.isinf(s), torch.zeros_like(s), s)

        return s, SS_wcs