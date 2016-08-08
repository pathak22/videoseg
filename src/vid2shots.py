"""
Divide a given video into multiple shots using the kernel temporal segmentation
library.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import os
from scipy.misc import imresize
from PIL import Image
from skimage import color
# from skimage.feature import hog
import numpy as np
import _init_paths  # noqa
import utils
from kts.cpd_auto import cpd_auto


def color_hist(im, colBins):
    """
    Get color histogram descriptors for RGB and LAB space.
    Input: im: (h,w,c): 0-255: np.uint8
    Output: descriptor: (colBins*6,)
    """
    assert im.ndim == 3 and im.shape[2] == 3, "image should be rgb"
    arr = np.concatenate((im, color.rgb2lab(im)), axis=2).reshape((-1, 6))
    desc = np.zeros((colBins * 6,), dtype=np.float)
    for i in range(3):
        desc[i * colBins:(i + 1) * colBins], _ = np.histogram(
            arr[:, i], bins=colBins, range=(0, 255))
        desc[i * colBins:(i + 1) * colBins] /= np.sum(
            desc[i * colBins:(i + 1) * colBins]) + (
            np.sum(desc[i * colBins:(i + 1) * colBins]) < 1e-4)
    i += 1
    desc[i * colBins:(i + 1) * colBins], _ = np.histogram(
        arr[:, i], bins=colBins, range=(0, 100))
    desc[i * colBins:(i + 1) * colBins] /= np.sum(
        desc[i * colBins:(i + 1) * colBins]) + (
        np.sum(desc[i * colBins:(i + 1) * colBins]) < 1e-4)
    for i in range(4, 6):
        desc[i * colBins:(i + 1) * colBins], _ = np.histogram(
            arr[:, i], bins=colBins, range=(-128, 127))
        desc[i * colBins:(i + 1) * colBins] /= np.sum(
            desc[i * colBins:(i + 1) * colBins]) + (
            np.sum(desc[i * colBins:(i + 1) * colBins]) < 1e-4)
    return desc


def compute_features(im, colBins):
    """
    Compute features of images: RGB histogram + SIFT
    im: (h,w,c): 0-255: np.uint8
    feat: (d,)
    """
    colHist = color_hist(im, colBins=colBins)
    # hogF = hog(
    #     color.rgb2gray(im), orientations=hogBins,
    #     pixels_per_cell=(hogCellSize, hogCellSize),
    #     cells_per_block=(int(np.sqrt(hogCells)),
    #                         int(np.sqrt(hogCells))),
    #     visualise=False)
    # return np.hstack((hogF, colHist))
    return colHist


def vid2shots(imSeq, maxShots=5, vmax=0.6, colBins=40):
    """
    Convert a given video into number of shots
    imSeq: (n,h,w,c): 0-255: np.uint8: RGB
    shotIdx: (k,): start Index of shot: 0-indexed
    shotScore: (k,): First change ../lib/kts/cpd_auto.py return value to
                     scores2 instead of costs (a bug)
    """
    X = np.zeros((imSeq.shape[0], compute_features(imSeq[0], colBins).size))
    print('Feature Matrix shape:', X.shape)
    for i in range(imSeq.shape[0]):
        X[i] = compute_features(imSeq[i], colBins)
    K = np.dot(X, X.T)
    shotIdx, _ = cpd_auto(K, maxShots - 1, vmax)
    shotIdx = np.concatenate(([0], shotIdx))
    return shotIdx


def parse_args():
    """
    Parse input arguments
    """
    import argparse
    parser = argparse.ArgumentParser(
        description='Creates a tracker using deepmatch and epicflow')
    parser.add_argument(
        '-out', dest='outdir',
        help='Directory to save output.',
        default=os.getenv("HOME") + '/local/data/trash/', type=str)
    parser.add_argument(
        '-imdir', dest='imdir',
        help='Directory containing video images. Will be read ' +
        'alphabetically. Default is random Imagenet train video.',
        default='', type=str)
    parser.add_argument(
        '-fgap', dest='frameGap',
        help='Gap between frames while running tracker. Default 0.',
        default=0, type=int)
    parser.add_argument(
        '-n', dest='maxShots',
        help='Max number of shots to break into. Default 5.',
        default=5, type=int)
    parser.add_argument(
        '-d', dest='colBins',
        help='Number of bins in RGBLAB histogram. Default 40. ',
        default=40, type=int)
    parser.add_argument(
        '-v', dest='vmax',
        help='Parameter for KTS, lower value means more clips. Default 0.6.',
        default=0.6, type=float)
    parser.add_argument(
        '-seed', dest='seed',
        help='Random seed for numpy and python.', default=2905, type=int)

    args = parser.parse_args()
    return args


def demo_images():
    """
    Input is the path of directory (imdir) containing images of a video
    """
    # Hard coded parameters
    maxSide = 400  # max length of longer side of Im
    lenSeq = 1e8  # longer seq will be shrinked between [lenSeq/2, lenSeq]

    # parse commandline parameters
    args = parse_args()
    np.random.seed(args.seed)
    if args.imdir == '':
        imagenetVideoList = '/mnt/vol/gfsai-local/ai-group/users/bharathh/' + \
                            'imagenet_videos/ILSVRC2015/ImageSets/VID/' + \
                            'train_10.txt'
        imagenetRoot = '/mnt/vol/gfsai-local/ai-group/users/bharathh/' + \
                    'imagenet_videos/ILSVRC2015/Data/VID/train/'
        with open(imagenetVideoList, 'r') as f:
            lines = f.readlines()
        imdirs = [x.strip().split(' ')[0] for x in lines]
        imdirs = imdirs[np.random.randint(len(imdirs))]
        args.imdir = os.path.join(imagenetRoot, imdirs)
        args.outdir = os.path.join(args.outdir, imdirs)

    # setup input directory
    print('InputDir: ', args.imdir)
    imPathList = utils.read_r(args.imdir, '*.*')
    if len(imPathList) < 2:
        print('Not enough images in image directory: \n%s' % args.imdir)
        return

    # setup output directory
    suffix = args.imdir.split('/')[-1]
    suffix = args.imdir.split('/')[-2] if suffix == '' else suffix
    args.outdir = args.outdir + '/' + suffix
    utils.mkdir_p(args.outdir)
    print('OutputDir: ', args.outdir)

    # load image sequence after adjusting frame gap and imsize
    frameGap = args.frameGap
    if frameGap <= 0 and len(imPathList) > lenSeq:
        frameGap = int(len(imPathList) / lenSeq)
    imPathList = imPathList[0:len(imPathList):frameGap + 1]
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
    print('Total Video Shape: ', imSeq.shape)

    # run the algorithm
    shotIdx = vid2shots(imSeq, maxShots=args.maxShots, vmax=args.vmax,
                            colBins=args.colBins)
    print('Total Shots: ', shotIdx.shape, shotIdx)
    np.save(args.outdir + '/shotIdx_%s.npy' % suffix, shotIdx)

    # save visual results
    from PIL import ImageDraw
    utils.rmdir_f(args.outdir + '/shots_%s/' % suffix)
    utils.mkdir_p(args.outdir + '/shots_%s/' % suffix)
    frameNo = 1
    shotNo = 0
    for i in range(imSeq.shape[0]):
        img = Image.fromarray(imSeq[i])
        draw = ImageDraw.Draw(img)
        if i in shotIdx:
            draw.text((100, 100), "New Shot Begins !!", (255, 255, 255))
            shotNo += 1
            frameNo = 1
        draw.text((10, 10), "Shot: %02d, Frame: %03d" % (shotNo, frameNo),
                    (255, 255, 255))
        img.save(
            args.outdir + '/shots_%s/' % suffix + imPathList[i].split('/')[-1])
        frameNo += 1
    import subprocess
    subprocess.call(
        ['tar', '-zcf', args.outdir + '/../shots_%s.tar.gz' % suffix,
            '-C', args.outdir + '/shots_%s/' % suffix, '.'])

    return


if __name__ == "__main__":
    demo_images()
