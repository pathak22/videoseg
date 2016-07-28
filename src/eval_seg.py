"""
To run:
python eval_seg.py -src ../datasets/gt_all.txt -target ../datasets/nlcgt_all.txt
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
        help='address of file containing path of images to be evaluated',
        type=str)
    parser.add_argument(
        '-ncl', dest='ncl',
        help='Num of classes including BG. Default 2.',
        default=2, type=int)
    parser.add_argument(
        '-patient', action='store_true',
        help='If you are patient enough to wait for full evaluation')
    parser.add_argument(
        '-notIgnore', action='store_true',
        help='Do not ignore any image')
    parser.add_argument(
        '-seed', dest='seed',
        help='Random seed for numpy and python.', default=2905, type=int)

    args = parser.parse_args()
    return args


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k],
                        minlength=n**2).reshape(n, n)


def compute_hist(srcImList, targetImList, ncl, patient, notIgnore):
    # read directory names
    with open(srcImList) as f:
        srcList = f.readlines()
    srcList = [line.rstrip('\n') for line in srcList]
    with open(targetImList) as f:
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
        im1 = (im1 > 0).astype(np.uint8)
        im2 = (im2 > 0).astype(np.uint8)
        if im2.size != im1.size:
            im1 = cv2.resize(im1, im2.shape, interpolation=cv2.INTER_NEAREST)
            if np.unique(im1).size > ncl:
                print('Integral resize bug !')
                exit(1)

        if im2.sum() > 0.9 * im2.size and not notIgnore:
            ignoreCount += 1
            continue

        hist += fast_hist(im1.flatten(), im2.flatten(), ncl)
        iu = np.zeros(ncl)
        if not patient:
            for i in range(ncl):
                iu[i] = hist[i, i] / (
                    hist[i].sum() + hist[:, i].sum() - hist[i, i])
            print('Image : ', count, ' ,  Name : ', srcIm.split('/')[-1],
                    ' , mean IU (till here) : ', np.nanmean(iu) * 100)
            sys.stdout.flush()
        count += 1
    print('>>> Total Images Ignored: ', ignoreCount)
    print('>>> Total Images Compared: ', len(srcList) - ignoreCount)
    return hist


def seg_tests(srcImList, targetImList, ncl, patient, notIgnore):
    print('>>>', datetime.now(), 'Begin seg tests')
    hist = compute_hist(srcImList, targetImList, ncl, patient, notIgnore)
    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    print('>>>', datetime.now(), 'overall accuracy: %.2f' % (acc * 100.))
    # per-class accuracy
    acc = np.zeros(ncl)
    for i in range(ncl):
        acc[i] = hist[i, i] / hist[i].sum()
    print('>>>', datetime.now(), 'mean accuracy: %.2f' %
            (np.nanmean(acc) *100.))
    # per-class IU
    iu = np.zeros(ncl)
    for i in range(ncl):
        iu[i] = hist[i, i] / (hist[i].sum() + hist[:, i].sum() - hist[i, i])
    print('>>>', datetime.now(), 'mean IU: %.2f' % (np.nanmean(iu) * 100))
    iu2 = [round(100 * elem, 1) for elem in iu]
    print('>>>', datetime.now(), 'per-class IU:', iu2)
    freq = hist.sum(1) / hist.sum()
    print('>>>', datetime.now(), 'fwavacc: %.2f' %
            ((freq[freq > 0] * iu[freq > 0]).sum() * 100.))
    print('')


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    seg_tests(args.srcImList, args.targetImList, args.ncl, args.patient,
                args.notIgnore)
