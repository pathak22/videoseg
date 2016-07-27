"""
This file implements following paper:
Video Segmentation by Non-Local Consensus Voting
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import sys
import time
from PIL import Image
import numpy as np
from scipy.misc import imresize
import _init_paths  # noqa
import utils
import nlc
import vid2shots
import crf


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
        '-numShards', dest='numShards',
        help='Number of shards for parallel running',
        default=1, type=int)
    parser.add_argument(
        '-shardId', dest='shardId',
        help='Shard to work on. Should range between 0 .. numShards-1',
        default=0, type=int)
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
    redirect = True  # redirecting to output file ? won't print status
    frameGap = 0  # 0 means adjusted automatically per shot (not per video)
    maxSide = 650  # max length of longer side of Im
    minShot = 10  # minimum shot length
    maxShot = 110  # longer shots will be shrinked between [maxShot/2, maxShot]
    binTh = 0.5  # final thresholding to obtain mask
    clear_blobs = True  # remove low energy blobs; uses binTh
    maxsp = 400
    iters = 100

    # For CRF:
    gtProb = 0.7
    posTh = binTh
    negTh = 0.3

    # For blob removal post CRF: more like salt-pepper noise removal
    bSize = 25  # 0 means not used, [0,1] relative, >=1 means absolute

    # parse commandline parameters
    args = parse_args()
    np.random.seed(args.seed)

    # read directory names
    with open(args.imdirFile) as f:
        imDirs = f.readlines()
    imDirs = [line.rstrip('\n') for line in imDirs]

    # keep only the current shard
    if args.shardId >= args.numShards:
        print('Give valid shard id which is less than numShards')
        exit(1)
    imDirs = [
        x for i, x in enumerate(imDirs) if i % args.numShards == args.shardId]
    print('NUM SHARDS: %03d,  SHARD ID: %03d,  CURRENT NUM VIDEOS: %03d\n\n' %
            (args.numShards, args.shardId, len(imDirs)))

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
        outVidNlc = args.baseOutdir + '/crfvid/'
        outVidCRF = args.baseOutdir + '/nlcvid/'

        utils.mkdir_p(outNlcIm)
        utils.mkdir_p(outNlcPy)
        utils.mkdir_p(outCrf)
        utils.mkdir_p(outIm)
        utils.mkdir_p(outVidNlc)
        utils.mkdir_p(outVidCRF)
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
                                outdir=outNlcPy, suffix=suffix,
                                redirect=redirect)
            if clear_blobs:
                maskSeq = nlc.remove_low_energy_blobs(maskSeq, binTh)
            np.save(outNlcPy + '/mask_%s.npy' % suffix, maskSeq)

            # run crf, run blob removal and save as images sequences
            sTime = time.time()
            crfSeq = np.zeros(maskSeq.shape, dtype=np.uint8)
            for i in range(maskSeq.shape[0]):
                mask = (maskSeq[i] > binTh).astype(np.uint8)
                Image.fromarray(mask).save(
                    outNlcIm + '/' +
                    imPathList1[i].split('/')[-1][:-4] + '.png')
                Image.fromarray(imSeq1[i]).save(
                    outIm + '/' + imPathList1[i].split('/')[-1])
                crfSeq[i] = crf.refine_crf(
                    imSeq1[i], maskSeq[i], gtProb=gtProb, posTh=posTh,
                    negTh=negTh)
                crfSeq[i] = utils.refine_blobs(crfSeq[i], bSize=bSize)
                Image.fromarray(crfSeq[i]).save(
                    outCrf + '/' +
                    imPathList1[i].split('/')[-1][:-4] + '.png')
                if not redirect:
                    sys.stdout.write(
                        'CRF, blob removal and saving: [% 5.1f%%]\r' %
                        (100.0 * float((i + 1) / maskSeq.shape[0])))
                    sys.stdout.flush()
            eTime = time.time()
            print('CRF, blob removal and saving images finished: %.2f s' %
                    (eTime - sTime))

            # save as video
            sTime = time.time()
            vidName = '_'.join(imdir.split('/')[3:]) + '_shot%d.avi' % (s + 1)
            utils.im2vid(outVidNlc + vidName, imSeq1, maskSeq)
            utils.im2vid(outVidCRF + vidName, imSeq1, crfSeq)
            eTime = time.time()
            print('Saving videos finished: %.2f s' % (eTime - sTime))

    return


if __name__ == "__main__":
    demo_images()
