"""
Adapted from pydensecrf/examples/inference.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
import _init_paths  # noqa
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary


def refine_crf(im, lb, gtProb=0.5, posTh=None, negTh=None, crfParams=0):
    """
    [ NOTE: Currently only supports n=2 i.e. FG/BG.
            For general n: Remove this line: `MAP*=1/MAP.max()` at the end]
    Convert a given video into number of shots
    im: (h,w,c): 0-255: np.uint8: RGB
    lb: (h,w): 0-255: int or float
        => int: it should have labels 1,..,n and must have one '0' special
        label which is not a label, but the special value indicating
        no-confidence region.
        => float: it should probabilities in [0,1]. Func will assign:
            label=2 to region with prob>=posTh
            label=1 to region with prob<=negTh
            label=0 to region with negTh<prob<posTh
    crfParams:
        value: 0: default crf params
        value: 1: deeplab crf params
        value: 2: ccnn crf params
    out: (h,w): np.uint8:
        For n=2: output labels are 0 and 1
                 0 means BG or uncertain (i.e. lb=0,1)
                 1 means FG (i.e. lb=2)
        For general n: Remove this line: `MAP*=1/MAP.max()` at the end
                 Then, label space is same as input i.e. in 0..n
    """
    # Hard coded CRF parameters
    iters = 5

    if crfParams == 1:
        # Deeplab Params
        xy_gauss = 19
        wt_gauss = 15
        xy_bilateral = 61
        rgb_bilateral = 10
        wt_bilateral = 35
    elif crfParams == 2:
        # untuned ccnn params
        xy_gauss = 6
        wt_gauss = 6
        xy_bilateral = 50
        rgb_bilateral = 4
        wt_bilateral = 5
    else:
        # Default Params
        xy_gauss = 3
        wt_gauss = 3
        xy_bilateral = 80
        rgb_bilateral = 13
        wt_bilateral = 10

    # take care of probability mask
    if lb.dtype == np.float32 or lb.dtype == np.float64:
        if posTh is None or negTh is None:
            print('For probability mask, labels are not given !')
            return
        lb1 = np.zeros(lb.shape, dtype=np.uint8)
        lb1[lb >= posTh] = 2
        lb1[lb <= negTh] = 1
        presentLb = np.unique(lb1)
        if presentLb.size < 3:
            if 2 not in presentLb:
                y, x = int(lb.shape[0] / 2), int(lb.shape[1] / 2)
                lb1[y - 1:y + 1, x - 1:x + 1] = 2  # center area as FG
            if 1 not in presentLb:
                lb1[0, :] = 1  # top row as BG
            if 0 not in presentLb:
                lb1[1, :] = 0  # second row: doesn't matter
        lb = lb1

    # convert to BGR
    im = np.ascontiguousarray(im[..., ::-1])

    # Compute the number of classes in the label image
    M = len(set(lb.flat))

    # Setup the 2D-CRF model
    d = dcrf.DenseCRF2D(im.shape[1], im.shape[0], M)

    # get unary potentials (neg log probability)
    U = compute_unary(lb, M)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(xy_gauss, xy_gauss), compat=wt_gauss)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(
        sxy=(xy_bilateral, xy_bilateral),
        srgb=(rgb_bilateral, rgb_bilateral, rgb_bilateral),
        rgbim=im, compat=wt_bilateral)

    # Do inference and compute map
    Q = d.inference(iters)
    MAP = np.argmax(Q, axis=0).astype('float32')
    MAP *= 1 / MAP.max()
    MAP = MAP.reshape(im.shape[:2])
    out = MAP.astype('uint8')

    return out


def parse_args():
    """
    Parse input arguments
    """
    import argparse
    parser = argparse.ArgumentParser(
        description='Creates a tracker using deepmatch and epicflow')
    parser.add_argument(
        '-out', dest='outIm',
        help='Path for output image.', type=str)
    parser.add_argument(
        '-inIm', dest='inIm',
        help='Path for input image.', type=str)
    parser.add_argument(
        '-inL', dest='inL',
        help='Path for input label.', type=str)
    parser.add_argument(
        '-gtProb', dest='gtProb',
        help='Ground Truth certainity for discrete labels. [0,1]. Default=0.5',
        default=0.5, type=float)
    parser.add_argument(
        '-seed', dest='seed',
        help='Random seed for numpy and python.', default=2905, type=int)

    args = parser.parse_args()
    return args


def demo_image():
    """
    Input is the path of directory (imdir) containing images of a video
    """
    from PIL import Image
    from skimage.segmentation import relabel_sequential
    args = parse_args()
    np.random.seed(args.seed)

    # NOTE: Both the codes below are same. However, in Philipp's examples:
    # actual labels are colormap output. Thus, if corresponding value in
    # anno1.png is 0, then label will be cmap[0].
    # By defualt cv2 reads cmap[0] and there is no way to read original value.
    # By defualt PIL reads val=0 and not the color mapped values.
    # Thus, the output is slightly different. However, code is correct.
    # They run just same on pngs with no colormap.

    # CV2 version of code:
    # import cv2
    # im = cv2.imread(args.inIm)
    # im = im[..., ::-1]
    # lb, _, _ = relabel_sequential(cv2.imread(args.inL, 0))
    # out = refine_crf(im, lb, gtProb=args.gtProb)
    # cv2.imwrite(args.outIm, out * 255)

    # PIL version of code:
    im = np.array(Image.open(args.inIm))
    lb, _, _ = relabel_sequential(np.array(Image.open(args.inL)))
    # uncomment while running for default examples in densecrf:
    # lb[lb==0] = 3; lb[lb==2] = 0; lb[lb==3] = 2
    out = refine_crf(im, lb, gtProb=args.gtProb)
    Image.fromarray(out * 255).save(args.outIm)

    return


if __name__ == "__main__":
    demo_image()
