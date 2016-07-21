"""
This file contains some common generic functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
import os


def vid2im(vidPath):
    """
    Convert video to image sequence.
    Output: imSeq: np array of shape (n,h,w,c): 0-255: np.uint8
    """
    import skvideo.io
    imSeq = skvideo.io.vread(vidPath, backend='ffmpeg', verbosity=0)
    return imSeq


def mkdir_p(path):
    """
    It creates directory recursively if it does not already exist
    """
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def rmdir_f(path):
    """
    It deletes directory recursively if it exist
    """
    import shutil
    shutil.rmtree(path, ignore_errors=True)


def read_r(indir, pattern):
    """
    It deletes directory recursively if it exist
    """
    import glob
    flist = sorted([y for (x, _, _) in os.walk(indir)
                        for y in glob.glob(os.path.join(x, pattern))])
    return flist


def draw_point_im(im, loc, col, sizeOut=10):
    """
    Draws point on the image at given locations.
    im.shape: (h,w,3): 0-255: np.uint8
    loc.shape: (n,2) or (2,) describing (y,x)
    col: (n,3) or (3,) where n is more than 1
    """
    loc = loc[None, :] if loc.ndim == 1 else loc
    col = col[None, :].repeat(loc.shape[0], axis=0) if col.ndim == 1 else col
    imNew = np.copy(im)
    sizeIn = sizeOut - 3
    for n in range(loc.shape[0]):
        imNew[loc[n, 0] - sizeOut:loc[n, 0] + sizeOut,
                loc[n, 1] - sizeOut:loc[n, 1] + sizeOut] = col[n]
        imNew[loc[
            n, 0] - sizeIn:loc[n, 0] + sizeIn,
            loc[n, 1] - sizeIn:loc[n, 1] + sizeIn
        ] = im[
            loc[n, 0] - sizeIn:loc[n, 0] + sizeIn,
            loc[n, 1] - sizeIn:loc[n, 1] + sizeIn]
    return imNew


def my_accumarray(indices, vals, size, func='plus', fill_value=0):
    """
    Implementing python equivalent of matlab accumarray.
    Taken from SDS repo: master/superpixel_representation.py#L36-L46
    """

    # get dictionary
    function_name_dict = {
        'plus': (np.add, 0.),
        'minus': (np.subtract, 0.),
        'times': (np.multiply, 1.),
        'max': (np.maximum, -np.inf),
        'min': (np.minimum, np.inf),
        'and': (np.logical_and, True),
        'or': (np.logical_or, False)}

    if func not in function_name_dict:
        raise KeyError('Function name not defined for accumarray')
    if np.isscalar(vals):
        if isinstance(indices, tuple):
            shape = indices[0].shape
        else:
            shape = indices.shape
        vals = np.tile(vals, shape)

    # get the function and the default value
    (function, value) = function_name_dict[func]

    # create an array to hold things
    output = np.ndarray(size)
    output[:] = value
    function.at(output, indices, vals)

    # also check whether indices have been used or not
    isthere = np.ndarray(size, 'bool')
    istherevals = np.ones(vals.shape, 'bool')
    (function, value) = function_name_dict['or']
    isthere[:] = value
    function.at(isthere, indices, istherevals)

    # fill things that were not used with fill value
    output[np.invert(isthere)] = fill_value
    return output
