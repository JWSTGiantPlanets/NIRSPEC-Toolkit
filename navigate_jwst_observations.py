#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to add navigation data backplanes (generated by JWSTSolarSystemPointing) to FITS 
files of JWST observations.

This script uses SPICE kernels, so you may need to change KERNEL_DIR or the values in
KERNEL_NAMES to work for your observations.

Example usage
-------------

>>> python3 navigate_jwst_observations.py data/stage3/*_s3d.fits

The output file paths will be of the format data/stage3_nav/*_s3d_nav.fits.
"""
import glob
import os
import pathlib
import sys
import warnings
import spiceypy as spice
import tqdm
from astropy.io import fits
import JWSTSolarSystemPointing as jssp

KERNEL_DIR = '~/spice'
KERNEL_NAMES = [
    'naif0012.tls',
    'pck00010.tpc',
    'de430.bsp',
    'jup310.bsp',
    'sat441.bsp',
    'jwst_*.bsp',
]


def main(*args) -> None:
    load_kernels()
    navigate_multiple(*args)


def navigate_multiple(*paths: str) -> None:
    print(f'{len(paths)} files to navigate...')
    with warnings.catch_warnings():
        # Hide warnings caused by NaN values in observations which include the
        # background sky
        warnings.filterwarnings('ignore', 'invalid value encountered in double_scalars')

        for path in tqdm.tqdm(paths):
            navigate_file(path)
    print(f'Saved backplane date for {len(paths)} files')


def navigate_file(path: str) -> None:
    path = os.path.abspath(path)
    geo = jssp.JWSTSolarSystemPointing(path)
    geo.full_fov()  # Generate backplanes
    with fits.open(path) as hdul:
        for key in geo.keys:
            data = geo.get_param(key)
            header = fits.Header()
            header.add_comment('Backplane generated by JWSTSolarSystemPointing')
            hdu = fits.ImageHDU(data=data, header=header, name=key)
            hdul.append(hdu)

        path_out = make_output_path(path)
        check_path(path_out)
        hdul.writeto(path_out, overwrite=True)


def make_output_path(path: str) -> str:
    directory_path, filename = os.path.split(path)
    root, directory = os.path.split(directory_path)
    directory = directory + '_nav'
    filename = filename.replace('.fits', '_nav.fits')
    return os.path.join(root, directory, filename)


def load_kernels() -> None:
    kerneldir = os.path.expanduser(KERNEL_DIR)

    for fn in KERNEL_NAMES:
        spice.furnsh(glob.glob(os.path.join(kerneldir, '**', fn), recursive=True))


def check_path(path) -> None:
    """
    Checks if file path's directory tree exists, and creates it if necessary.

    Assumes path is to a file if `os.path.split(path)[1]` contains '.',
    otherwise assumes path is to a directory.

    Parameters
    ----------
    path : str
        Path to directory to check.
    """
    if os.path.isdir(path):
        return
    if '.' in os.path.split(path)[1]:
        path = os.path.split(path)[0]
        if os.path.isdir(path):
            return
    if path == '':
        return
    print('Creating directory path "{}"'.format(path))
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
