"""
This file implements following paper:
Video Segmentation by Non-Local Consensus Voting
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
from PIL import Image
import numpy as np
from scipy.misc import imresize
import _init_paths  # noqa
import utils
import pydensecrf.densecrf as dcrf
import nlc
import vid2shots


def parse_args():
    """
    Parse input arguments
    """
    import argparse
    parser = argparse.ArgumentParser(
        description='Foreground Segmentation using Non-Local Consensus')
    parser.add_argument(
        '-out', dest='baseOutdir',
        help='Base directory to save output.',
        type=str)
    parser.add_argument(
        '-in', dest='imdirFile',
        help='Addresses of file containing list of video image directories.' +
        ' Each imdir will be read alphabetically for video image sequence.',
        type=str)
    parser.add_argument(
        '-seed', dest='seed',
        help='Random seed for numpy and python.', default=2905, type=int)

    args = parser.parse_args()
    return args


def demo_images():
    """
    Input is the path of file containing list of directories of video images.
    """
    # Hard coded parameters
    # VSB has 121 images per video
    # FBMS has 100-250 images per video
    # DAVIS has 60-70 images per video

    # For Shot:
    maxShots = 10
    vmax = 0.2
    colBins = 40

    # For NLC:
    frameGap = 0  # 0 means adjusted automatically per shot (not per video)
    maxSide = 650  # max length of longer side of Im
    minShot = 10  # minimum shot length
    maxShot = 110  # longer shots will be shrinked between [maxShot/2, maxShot]
    binTh = 0.5  # final thresholding to obtain mask
    clear_blobs = True  # remove low energy blobs; uses binTh
    maxsp = 400
    iters = 100

    # parse commandline parameters
    args = parse_args()
    np.random.seed(args.seed)

    # read directory names
    with open(args.imdirFile) as f:
        imDirs = f.readlines()
    imDirs = [line.rstrip('\n') for line in imDirs]

    for imdir in imDirs:
        # setup input directory
        print('-------------------------------------------------------------\n')
        print('Video InputDir: ', imdir)
        imPathList = utils.read_r(imdir, '*.jpg')
        if len(imPathList) < 2:
            print('Not enough images in image directory: \n%s' % imdir)
            print('Continuing to next one ...')
            continue

        # setup output directory
        suffix = imdir.split('/')[-1]
        suffix = imdir.split('/')[-2] if suffix == '' else suffix
        outNlcIm = args.baseOutdir.split('/') + ['nlcim'] + imdir.split('/')[3:]
        outNlcPy = args.baseOutdir.split('/') + ['nlcpy'] + imdir.split('/')[3:]
        outCrf = args.baseOutdir.split('/') + ['crf'] + imdir.split('/')[3:]
        outIm = args.baseOutdir.split('/') + ['im'] + imdir.split('/')[3:]

        outNlcIm = '/'.join(outNlcIm)
        outNlcPy = '/'.join(outNlcPy)
        outCrf = '/'.join(outCrf)
        outIm = '/'.join(outIm)

        utils.mkdir_p(outNlcIm)
        utils.mkdir_p(outNlcPy)
        utils.mkdir_p(outCrf)
        utils.mkdir_p(outIm)
        print('Video OutputDir: ', outNlcIm)

        # resize images if needed
        h, w, c = np.array(Image.open(imPathList[0])).shape
        frac = min(min(1. * maxSide / h, 1. * maxSide / w), 1.0)
        if frac < 1.0:
            h, w, c = imresize(np.array(Image.open(imPathList[0])), frac).shape
        imSeq = np.zeros((len(imPathList), h, w, c), dtype=np.uint8)
        for i in range(len(imPathList)):
            if frac < 1.0:
                imSeq[i] = imresize(np.array(Image.open(imPathList[i])), frac)
            else:
                imSeq[i] = np.array(Image.open(imPathList[i]))

        # First run shot detector
        shotIdx = vid2shots.vid2shots(imSeq, maxShots=maxShots, vmax=vmax,
                                        colBins=colBins)
        print('Total Shots: ', shotIdx.shape, shotIdx)
        np.save(outNlcPy + '/shotIdx_%s.npy' % suffix, shotIdx)

        # Adjust frameGap per shot, and then run NLC per shot
        for s in range(shotIdx.shape[0]):
            suffix = suffix + '_shot%d' % (s + 1)

            shotS = shotIdx[s]  # 0-indexed, included
            shotE = imSeq.shape[0] if s == shotIdx.shape[0] - 1 \
                else shotIdx[s + 1]  # 0-indexed, excluded
            shotL = shotE - shotS
            if shotL < minShot:
                continue

            if frameGap <= 0 and shotL > maxShot:
                frameGap = int(shotL / maxShot)
            imPathList1 = imPathList[shotS:shotE:frameGap + 1]
            imSeq1 = imSeq[shotS:shotE:frameGap + 1]

            print('\nShot: %d, Shape: ' % (s + 1), imSeq1.shape)
            maskSeq = nlc.nlc(imSeq1, maxsp=maxsp, iters=iters,
                                outdir=outNlcPy, suffix=suffix, redirect=True)
            if clear_blobs:
                maskSeq = nlc.remove_low_energy_blobs(maskSeq, binTh)
            np.save(outNlcPy + '/mask_%s.npy' % suffix, maskSeq)

            for i in range(maskSeq.shape[0]):
                mask = (maskSeq[i] > binTh).astype(np.uint8)
                Image.fromarray(mask).save(
                    outNlcIm + '/' +
                    imPathList1[i].split('/')[-1][:-4] + '.png')
                Image.fromarray(imSeq1[i]).save(
                    outIm + '/' + imPathList1[i].split('/')[-1])

    return


if __name__ == "__main__":
    demo_images()
