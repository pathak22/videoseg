from __future__ import print_function, division

import cv2
from subprocess import call
from shutil import copyfile
import os
import numpy as np


def run_deepmatch(imname1, imname2):
    command = os.getenv("HOME") + '/fbcode/_bin/experimental/' + \
        'deeplearning/dpathak/video-processing/deepmatch/deepmatch'
    call([command, imname1, imname2,
            '-out', os.getenv("HOME") + '/local/data/trash/tmp.txt',
            '-downscale', '2'])
    with open(os.getenv("HOME") + '/local/data/trash/tmp.txt', 'r') as f:
        lines = f.readlines()

    lines = [x.strip().split(' ') for x in lines]
    vals = np.array([[float(y) for y in x] for x in lines])
    x = ((vals[:, 0] - 8.) / 16.).astype(int)
    y = ((vals[:, 1] - 8.) / 16.).astype(int)
    U = np.zeros((int(np.max(y)) + 1, int(np.max(x)) + 1))
    U[(y, x)] = vals[:, 2] - vals[:, 0]
    V = np.zeros((int(np.max(y)) + 1, int(np.max(x)) + 1))
    V[(y, x)] = vals[:, 3] - vals[:, 1]

    img1 = cv2.imread(imname1)
    U1 = cv2.resize(U, (img1.shape[1], img1.shape[0]))
    V1 = cv2.resize(V, (img1.shape[1], img1.shape[0]))

    mag, ang = cv2.cartToPolar(U1, V1)
    print(np.max(mag))
    hsv = np.zeros_like(img1)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def get_examples(outdir=os.getenv("HOME") + '/local/data/trash/'):
    with open('/mnt/vol/gfsai-local/ai-group/users/bharathh/imagenet_videos' +
                '/ILSVRC2015/ImageSets/VID/train_10.txt', 'r') as f:
        lines = f.readlines()

    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    imdirs = [x.strip().split(' ')[0] for x in lines]
    rootdir = '/mnt/vol/gfsai-local/ai-group/users/bharathh/' + \
                'imagenet_videos/ILSVRC2015/Data/VID/train/'
    imdirs = [os.path.join(rootdir, x) for x in imdirs]
    ri = np.random.choice(len(imdirs), 4, False)
    imdirs = [imdirs[i] for i in ri]
    for i, d in enumerate(imdirs):
        print(i)
        files = os.listdir(d)
        files.sort()
        print(files[:2])
        chosenids = [0, 10]
        bgr = run_deepmatch(os.path.join(d, files[chosenids[0]]),
                                        os.path.join(d, files[chosenids[1]]))
        copyfile(os.path.join(d, files[chosenids[0]]),
                                os.path.join(outdir, '{:d}_src.JPEG'.format(i)))
        copyfile(os.path.join(d, files[chosenids[1]]),
                                os.path.join(outdir, '{:d}_dst.JPEG'.format(i)))
        cv2.imwrite(os.path.join(outdir, '{:d}_flow.jpg'.format(i)), bgr)
