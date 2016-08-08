"""
To run: see eval_seg_batch.sh
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import cv2
import numpy as np
import sys
import os
from PIL import Image
from datetime import datetime


def parse_args():
    """
    Parse input arguments
    """
    import argparse
    parser = argparse.ArgumentParser(
        description='Calculate IoU between two list of images')
    parser.add_argument(
        '-target', dest='targetImList',
        help='address of file containing path of images to be evaluated',
        type=str)
    parser.add_argument(
        '-src', dest='srcImList',
        help='address of file containing path of gt images to be evaluated',
        type=str)
    parser.add_argument(
        '-ncl', dest='ncl',
        help='Num of classes including BG. Default 2.',
        default=2, type=int)
    parser.add_argument(
        '-iouL', dest='iouL',
        help='Lower bound on image FG IoU [0-100]% for ignoring it',
        default=0, type=float)
    parser.add_argument(
        '-fgL', dest='fgL',
        help='Lower bound on size of image FG [0-100]% for ignoring it',
        default=0, type=float)
    parser.add_argument(
        '-fgU', dest='fgU',
        help='Upper bound on size of image FG [0-100]% for ignoring it',
        default=100, type=float)
    parser.add_argument(
        '-patient', action='store_true',
        help='If you are patient enough to wait for full evaluation')
    parser.add_argument(
        '-seed', dest='seed',
        help='Random seed for numpy and python.', default=2905, type=int)

    args = parser.parse_args()
    return args


def fast_hist(a, b, n):
    # a: ground truth label vector
    # b: predicted label vector
    # Think of it as 2-D histogram, where each cell is numbered uniquely in
    # [0,n^2-1] row-wise, i.e. (0,0), (0,1), (0,2) ... (n-1,n-1)
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k],
                        minlength=n**2).reshape(n, n)


def compute_hist(args):
    # read directory names
    ncl = args.ncl
    with open(args.srcImList) as f:
        srcList = f.readlines()
    srcList = [line.rstrip('\n') for line in srcList]
    with open(args.targetImList) as f:
        targetList = f.readlines()
    targetList = [line.rstrip('\n') for line in targetList]

    count = 1
    ignoreCount = 0
    hist = np.zeros((ncl, ncl))
    for ind in range(len(srcList)):
        srcIm = srcList[ind]
        targetIm = targetList[ind]
        if not os.path.isfile(targetIm):
            print('Missing target file!! => ', targetIm)
            print('exiting..')
            exit(1)
        if targetIm.split('/')[-1][:-4] not in srcIm.split('/')[-1]:
            print(targetIm.split('/')[-1][:-4], srcIm.split('/')[-1])
            print('Mismatch error !')
            exit(1)

        im1 = np.array(Image.open(srcIm))
        im2 = np.array(Image.open(targetIm))
        im1 = (im1 > 0).astype(np.uint8)  # take all instances in FG
        im2 = (im2 > 0).astype(np.uint8)  # take all instances in FG
        if im2.size != im1.size:
            im1 = cv2.resize(im1, im2.shape, interpolation=cv2.INTER_NEAREST)
            if np.unique(im1).size > ncl:
                print('Integral resize bug !')
                exit(1)

        if (im2.sum() <= args.fgL * 0.01 * im2.size or
                im2.sum() > args.fgU * 0.01 * im2.size):
            ignoreCount += 1
            continue

        iu = np.zeros(ncl)
        imHist = fast_hist(im1.flatten(), im2.flatten(), ncl)
        if args.iouL > 0:
            for i in range(ncl):
                iu[i] = imHist[i, i] / (
                    imHist[i].sum() + imHist[:, i].sum() - imHist[i, i])
            if iu[1] * 100 < args.iouL:
                ignoreCount += 1
                continue

        hist += imHist
        if not args.patient:
            for i in range(ncl):
                iu[i] = hist[i, i] / (
                    hist[i].sum() + hist[:, i].sum() - hist[i, i])
            print('Image : ', count, ' ,  Name : ', srcIm.split('/')[-1],
                    ' , mean IU (till here) : ', np.nanmean(iu) * 100)
            sys.stdout.flush()
        count += 1
    return hist, ignoreCount, len(srcList) - ignoreCount


def seg_tests(args):
    ncl = args.ncl
    print('>>>', datetime.now(), 'Begin seg tests')
    hist, ignore, comp = compute_hist(args)
    # overall accuracy
    acc1 = 100. * np.diag(hist).sum() / hist.sum()
    print('>>>', datetime.now(), 'overall accuracy: %.2f' % acc1)
    # per-class accuracy
    acc2 = np.zeros(ncl)
    for i in range(ncl):
        acc2[i] = 100. * hist[i, i] / hist[i].sum()
    print('>>>', datetime.now(), 'mean accuracy: %.2f' % np.nanmean(acc2))
    # per-class precision = TP / (TP+FP)
    precision = np.zeros(ncl)
    for i in range(ncl):
        precision[i] = 100. * hist[i, i] / hist[:, i].sum()
    # per-class recall = TP / (TP+FN)
    recall = np.zeros(ncl)
    for i in range(ncl):
        recall[i] = 100. * hist[i, i] / hist[i, :].sum()
    # per-class IU
    iu = np.zeros(ncl)
    for i in range(ncl):
        iu[i] = hist[i, i] / (hist[i].sum() + hist[:, i].sum() - hist[i, i])
    print('>>>', datetime.now(), 'mean IU: %.2f' % (np.nanmean(iu) * 100))
    iu2 = [round(100 * elem, 2) for elem in iu]
    print('>>>', datetime.now(), 'per-class IU:', iu2)
    # take frequency into account
    freq = hist.sum(1) / hist.sum()
    print('>>>', datetime.now(), 'fwavacc: %.2f' %
            ((freq[freq > 0] * iu[freq > 0]).sum() * 100.))
    print('>>> Total Images Ignored: ', ignore)
    print('>>> Total Images Compared: ', comp - ignore)
    # pretty print
    print('Pretty Output for Sheet:')
    print('Ignored # Im, Compared # Im, FG Lower, FG Upper, IoU LowerTh, ' +
            'mean Acc, overall Acc, IoU mean, IoU FG, FG Precision,' +
            ' FG Recall, IoU BG, BG Precision, BG Recall')
    print(
        '=SPLIT(\"',
        '%d %d %.1f %.1f %.1f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' %
        (ignore, comp, args.fgL, args.fgU, args.iouL, np.nanmean(acc2), acc1,
            np.nanmean(iu) * 100, iu2[1], precision[1], recall[1],
            iu2[0], precision[0], recall[0]),
        '\",\" \")')


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    seg_tests(args)
