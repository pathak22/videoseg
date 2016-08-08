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
        '-doload', dest='doload',
        help='load from .npy files already existing run. 0 or 1. Default 0.',
        default=0, type=int)
    parser.add_argument(
        '-dosave', dest='dosave',
        help='save .npy files at each important step Takes lot of space.' +
        ' 0 or 1. Default 0.',
        default=0, type=int)
    parser.add_argument(
        '-crfParams', dest='crfParams',
        help='CRF Params: default=0, deeplab=1, ccnn=2. Default is 0.',
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
    maxShots = 5
    vmax = 0.6
    colBins = 40

    # For NLC:
    redirect = True  # redirecting to output file ? won't print status
    frameGap = 0  # 0 means adjusted automatically per shot (not per video)
    maxSide = 650  # max length of longer side of Im
    minShot = 10  # minimum shot length
    maxShot = 110  # longer shots will be shrinked between [maxShot/2, maxShot]
    binTh = 0.7  # final thresholding to obtain mask
    clearVoteBlobs = True  # remove small blobs in consensus vote; uses binTh
    relEnergy = binTh - 0.1  # relative energy in consensus vote blob removal
    clearFinalBlobs = True  # remove small blobs finally; uses binTh
    maxsp = 400
    iters = 50

    # For CRF:
    gtProb = 0.7
    posTh = binTh
    negTh = 0.4

    # For blob removal post CRF: more like salt-pepper noise removal
    bSize = 25  # 0 means not used, [0,1] relative, >=1 means absolute

    # parse commandline parameters
    args = parse_args()
    np.random.seed(args.seed)
    doload = bool(args.doload)
    dosave = bool(args.dosave)

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
        numTries = 0
        totalTries = 2
        sleepTime = 60
        while numTries < totalTries:
            imPathList = utils.read_r(imdir, '*.jpg')
            # imPathList = imPathList + utils.read_r(imdir, '*.bmp')
            if len(imPathList) < 1:
                print('Failed to load ! Trying again in %d seconds' % sleepTime)
                numTries += 1
                time.sleep(sleepTime)  # delays for x seconds
            else:
                break
        if len(imPathList) < 2:
            print('Not enough images in image directory: \n%s' % imdir)
            # print('Continuing to next one ...')
            # continue
            assert False, 'Image directory does not exist !!'

        # setup output directory
        suffix = imdir.split('/')[-1]
        suffix = imdir.split('/')[-2] if suffix == '' else suffix
        outNlcIm = args.baseOutdir.split('/') + \
            ['nlcim', 'shard%03d' % args.shardId] + imdir.split('/')[3:]
        outNlcPy = args.baseOutdir.split('/') + ['nlcpy'] + imdir.split('/')[3:]
        outCrf = args.baseOutdir.split('/') + \
            ['crfim', 'shard%03d' % args.shardId] + imdir.split('/')[3:]
        outIm = args.baseOutdir.split('/') + \
            ['im', 'shard%03d' % args.shardId] + imdir.split('/')[3:]

        outNlcIm = '/'.join(outNlcIm)
        outNlcPy = '/'.join(outNlcPy)
        outCrf = '/'.join(outCrf)
        outIm = '/'.join(outIm)
        outVidNlc = args.baseOutdir + '/nlcvid/'
        outVidCRF = args.baseOutdir + '/crfvid/'

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
        if not doload:
            shotIdx = vid2shots.vid2shots(imSeq, maxShots=maxShots, vmax=vmax,
                                            colBins=colBins)
            if dosave:
                np.save(outNlcPy + '/shotIdx_%s.npy' % suffix, shotIdx)
        else:
            shotIdx = np.load(outNlcPy + '/shotIdx_%s.npy' % suffix)
        print('Total Shots: ', shotIdx.shape, shotIdx)

        # Adjust frameGap per shot, and then run NLC per shot
        for s in range(shotIdx.shape[0]):
            suffixShot = suffix + '_shot%d' % (s + 1)

            shotS = shotIdx[s]  # 0-indexed, included
            shotE = imSeq.shape[0] if s == shotIdx.shape[0] - 1 \
                else shotIdx[s + 1]  # 0-indexed, excluded
            shotL = shotE - shotS
            if shotL < minShot:
                continue

            frameGapLocal = frameGap
            if frameGapLocal <= 0 and shotL > maxShot:
                frameGapLocal = int(shotL / maxShot)
            imPathList1 = imPathList[shotS:shotE:frameGapLocal + 1]
            imSeq1 = imSeq[shotS:shotE:frameGapLocal + 1]

            print('\nShot: %d, Shape: ' % (s + 1), imSeq1.shape)
            if not doload:
                maskSeq = nlc.nlc(imSeq1, maxsp=maxsp, iters=iters,
                                    outdir=outNlcPy, suffix=suffixShot,
                                    clearBlobs=clearVoteBlobs, binTh=binTh,
                                    relEnergy=relEnergy,
                                    redirect=redirect, doload=doload,
                                    dosave=dosave)
                if clearFinalBlobs:
                    maskSeq = nlc.remove_low_energy_blobs(maskSeq, binTh)
                if dosave:
                    np.save(outNlcPy + '/mask_%s.npy' % suffixShot, maskSeq)
            if doload:
                maskSeq = np.load(outNlcPy + '/mask_%s.npy' % suffixShot)

            # run crf, run blob removal and save as images sequences
            sTime = time.time()
            crfSeq = np.zeros(maskSeq.shape, dtype=np.uint8)
            for i in range(maskSeq.shape[0]):
                # save soft score as png between 0 to 100.
                # Use binTh*100 to get FG in later usage.
                mask = (maskSeq[i] * 100).astype(np.uint8)
                Image.fromarray(mask).save(
                    outNlcIm + '/' +
                    imPathList1[i].split('/')[-1][:-4] + '.png')
                Image.fromarray(imSeq1[i]).save(
                    outIm + '/' + imPathList1[i].split('/')[-1])
                crfSeq[i] = crf.refine_crf(
                    imSeq1[i], maskSeq[i], gtProb=gtProb, posTh=posTh,
                    negTh=negTh, crfParams=args.crfParams)
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
            utils.im2vid(outVidNlc + vidName, imSeq1,
                            (maskSeq > binTh).astype(np.uint8))
            utils.im2vid(outVidCRF + vidName, imSeq1, crfSeq)
            eTime = time.time()
            print('Saving videos finished: %.2f s' % (eTime - sTime))

    # Tarzip the results of this shard and delete the individual files
    import subprocess
    for i in ['im', 'crfim', 'nlcim']:
        tarDir = args.baseOutdir + '/%s/shard%03d' % (i, args.shardId)
        subprocess.call(['tar', '-zcf', tarDir + '.tar.gz', '-C',
                        tarDir, '.'])
        utils.rmdir_f(tarDir)

    return


if __name__ == "__main__":
    demo_images()
