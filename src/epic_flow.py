from __future__ import print_function, division

import numpy as np
import cv2
import os
from subprocess import call
from scipy.signal import convolve2d


def read_flo(filename):
    with open(filename, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                print('Magic number incorrect. Invalid .flo file')
            else:
                w = np.fromfile(f, np.int32, count=1)
                h = np.fromfile(f, np.int32, count=1)
                print('Reading %d x %d flo file' % (w, h))
                data = np.fromfile(f, np.float32, count=2 * w * h)
                # Reshape data into 3D array (columns, rows, bands)
                data2D = np.resize(data, (h, w, 2))

    return data2D


def compute_flow(impath1, impath2, outdir,
                    fbcodepath=os.getenv("HOME") + '/fbcode'):
    stem = os.path.splitext(os.path.basename(impath1))[0]
    deepmatch_cmd = os.path.join(fbcodepath,
                                    '_bin/experimental/deeplearning/dpathak' +
                                    '/video-processing/deepmatch/deepmatch')
    call([deepmatch_cmd, impath1, impath2, '-out',
                os.path.join(outdir, stem + '_sparse.txt'), '-downscale', '2'])
    img1 = cv2.imread(impath1).astype(float)
    M = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.float32)
    filt = np.array([[1., -1.]]).reshape((1, -1))
    for c in range(3):
        gx = convolve2d(img1[:, :, c], filt, mode='same')
        gy = convolve2d(img1[:, :, c], filt.T, mode='same')
        M = M + gx**2 + gy**2

    M = M / np.max(M)
    with open(os.path.join(outdir, '_edges.bin'), 'w') as f:
        M.tofile(f)

    epicflow_command = os.path.join(fbcodepath,
                                    '_bin/experimental/deeplearning/dpathak' +
                                    '/video-processing/epicflow/epicflow')
    call([epicflow_command, impath1, impath2,
                os.path.join(outdir, '_edges.bin'),
                os.path.join(outdir, stem + '_sparse.txt'),
                os.path.join(outdir, 'flow.flo')])

    flow = read_flo(os.path.join(outdir, 'flow.flo'))
    hsv = np.zeros_like(img1).astype(np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0].astype(float),
                                flow[..., 1].astype(float))
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 0] = ang * 180 / np.pi / 2
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join(outdir, stem + '_flow.png'), bgr)


def save_flow_all(ranges, imdir, outdir):
    for i in range(len(ranges[0])):
        start = ranges[0][i]
        stop = ranges[1][i]
        shot_outdir = os.path.join(outdir, '{:d}'.format(i))
        if not os.path.isdir(shot_outdir):
            os.mkdir(shot_outdir)

        for j in range(start, stop, 20):
            src = '{}/{:06d}.JPEG'.format(imdir, j)
            dst = '{}/{:06d}.JPEG'.format(imdir, j + 1)
            print(src)
            print(dst)
            compute_flow(src, dst, shot_outdir)
            print('{:d}:{:d}/{:d}'.format(i, j + 1 - start, stop - start))
