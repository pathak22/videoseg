"""
This file contains some common generic functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
import os
from scipy import ndimage


def vid2im(vidPath):
    """
    Convert video to image sequence.
    Output: imSeq: np array of shape (n,h,w,c): 0-255: np.uint8: RGB
    """
    # skivideo is unreliable ! Don't use
    # import skvideo.io
    # imSeq = skvideo.io.vread(vidPath, backend='ffmpeg', verbosity=0)
    import cv2
    vidcap = cv2.VideoCapture()
    vidcap.open(vidPath)
    if not vidcap.isOpened():
        return None
    imSeq = []
    notdone = True
    while notdone:
        notdone, frame = vidcap.read()
        if notdone:
            imSeq.append(frame[np.newaxis, :, :, :])
    imSeq = np.concatenate(imSeq)
    imSeq = imSeq[..., ::-1]
    return imSeq


def im2vid(vidPath, imSeq, maskSeq=None):
    """
    Convert an image sequence to video and write to vidPath. If mask is given,
        then it generates a nice mask laid over video.
    Input: imSeq: np array of shape (n,h,w,c): 0-255: np.uint8: RGB
    maskSeq: same size and shape as imSeq: {0,1}: 0=BG, 1=FG: uint8
    """
    import cv2
    writer = cv2.VideoWriter(
        vidPath, cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), 10,
        (imSeq[0].shape[1], imSeq[0].shape[0]))
    if not writer.isOpened():
        print('Video could not be written. Some bug!')
        return None
    for i in range(imSeq.shape[0]):
        if maskSeq is not None:
            mask = maskSeq[i]
            grayscaleimage = cv2.cvtColor(
                imSeq[i, :, :, ::-1].copy(), cv2.COLOR_BGR2GRAY)
            maskedimage = np.zeros(imSeq[i].shape, dtype=np.uint8)
            for c in range(3):
                maskedimage[:, :, c] = grayscaleimage / 2 + 127
            maskedimage[mask.astype(np.bool), :2] = 0
        else:
            maskedimage = imSeq[i, :, :, ::-1]
        writer.write(maskedimage)
    writer.release()
    writer = None
    return


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
    It reads filname containing input pattern recursively if indir exists
    """
    import glob
    flist = sorted([y for (x, _, _) in os.walk(indir)
                        for y in glob.glob(os.path.join(x, pattern))])
    return flist


def draw_point_im(im, loc, col, sizeOut=10):
    """
    Draws point on the image at given locations.
    im.shape: (h,w,3): 0-255: np.uint8: RGB
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


def refine_blobs(maskSeq, bSize=0.6):
    """
    --> Adapted from src/nlc.py:remove_low_energy_blobs()
    Input:
        maskSeq: (n, h, w) or (h, w): int. FG=1, BG=0.
        bSize:
            (0,1): min relative size of FG blobs to keep compared to largest one
            >= 1: minimum absolute size of blobs to keep
            <= 0: not do anything and return
    Output:
        maskSeq: same as input
    """
    if bSize <= 0:
        return maskSeq

    if maskSeq.ndim < 3:
        maskSeq = maskSeq[None, ...]

    for i in range(maskSeq.shape[0]):
        mask = maskSeq[i]
        if np.sum(mask) == 0:
            continue
        sp1, num = ndimage.label(mask)  # 0 in sp1 is same as 0 in mask i.e. BG
        count = my_accumarray(sp1, np.ones(sp1.shape), num + 1, 'plus')
        sizeLargestBlob = np.max(count[1:])
        th = bSize if bSize >= 1 else bSize * sizeLargestBlob
        destroyFG = count[1:] <= th
        destroyFG = np.concatenate(([False], destroyFG))
        maskSeq[i][destroyFG[sp1]] = 0

    if maskSeq.shape[0] == 1:
        return maskSeq[0]
    return maskSeq


def my_accumarray(indices, vals, size, func='plus', fill_value=0):
    """
    Implementing python equivalent of matlab accumarray.
    Taken from SDS repo: master/superpixel_representation.py#L36-L46
        indices: must be a numpy array (any shape)
        vals: numpy array of same shape as indices or a scalar
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
