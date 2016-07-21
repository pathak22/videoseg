"""
This file performs deepmatch tracking. That is, it computes deepmatch
correspondeces between pairwise images of a given video and then combine
them to obtain long term tracks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
from PIL import Image
import numpy as np
import os
import sys
import utils


def frame_homography(totalPts, homTh):
    """
    Filter foreground points i.e. the outlier points found by fitting
    homography using RANSAC
    Input:
        totalPts: (numAllPoints, 4): x0, y0, x1, y1
        fgPts: (numAllPoints, 4): x0, y0, x1, y1
    """
    if totalPts.ndim != 2 or totalPts.shape[0] < 8 or homTh < 0:
        return totalPts

    import cv2
    p1 = totalPts[:, :2].astype('float')
    p2 = totalPts[:, 2:4].astype('float')
    _, status = cv2.findHomography(
        p1, p2, cv2.RANSAC, ransacReprojThreshold=homTh)
    fgPts = totalPts[status[:, 0] == 0, :]
    return fgPts


def shot_homography(shotTracks, homTh):
    """
    Filter foreground points i.e. the outlier points found by fitting
    homography using RANSAC
    Input:
        shotTracks: (numFrames, numAllPoints, 2)
        fgTracks: (numFrames, numForegroundPoints, 2)
    """
    if shotTracks.ndim < 3 or shotTracks.shape[0] < 2 or homTh < 0:
        return shotTracks

    import cv2
    status = 1
    for i in range(1, shotTracks.shape[0]):
        if shotTracks[i - 1, 0, 2] > -1000:
            p1 = shotTracks[i - 1, :, 2:].astype('float')
        else:
            p1 = shotTracks[i - 1, :, :2].astype('float')
        p2 = shotTracks[i, :, :2].astype('float')
        _, new_status = cv2.findHomography(
            p1, p2, cv2.RANSAC, ransacReprojThreshold=homTh)
        status = new_status * status

    fgTracks = shotTracks[:, status[:, 0] == 0, :]
    print(shotTracks.shape[0], shotTracks.shape[1], fgTracks.shape[1])
    return fgTracks


def run_epicFlow_pair(impath1, impath2, flowdir, deepmatchfile, fNo=1,
                        vizFlow=False):
    """
    Run EpicFlow code between two images
    """

    def read_flo(filename):
        """
        Function for reading output file of epicflow
        """
        with open(filename, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                print('Magic number incorrect. Invalid .flo file')
            else:
                w = np.fromfile(f, np.int32, count=1)
                h = np.fromfile(f, np.int32, count=1)
                # print('Reading %d x %d flo file' % (w, h))
                data = np.fromfile(f, np.float32, count=2 * w * h)
                # Reshape data into 3D array (columns, rows, bands)
                data2D = np.resize(data, (h, w, 2))
        return data2D

    # set file names
    edgefile = flowdir + 'edges_%04d.bin' % fNo
    flofile = flowdir + 'flow_%04d.flo' % fNo
    flowMatchFile = flowdir + 'match_%04d.txt' % fNo

    # # compute edge map
    # im1 = np.array(Image.open(impath1)).astype(float)
    # from scipy.ndimage.filters import gaussian_filter
    # im1 = gaussian_filter(im1, 5)
    #
    # # compute edge map using image gradients
    # from scipy.signal import convolve2d
    # M = np.zeros((im1.shape[0], im1.shape[1]), dtype=np.float32)
    # filt = np.array([[1., -1.]]).reshape((1, -1))
    # for c in range(3):
    #     gx = convolve2d(im1[:, :, c], filt, mode='same')
    #     gy = convolve2d(im1[:, :, c], filt.T, mode='same')
    #     M = M + gx**2 + gy**2
    # M = np.sqrt(M)
    # M = M / np.max(M)
    # M = M
    # M = M.astype(np.float32)
    # # compute edge map using canny
    # # from skimage import feature, color
    # # M = feature.canny(color.rgb2gray(im1), sigma=21)
    # M.tofile(edgefile, "")
    #
    # # make blocking subprocess call to deepmatch command
    # # output format: x0, y0, x1, y1, score, index
    # import subprocess
    # command = os.getenv("HOME") + '/fbcode/_bin/experimental/' + \
    #     'deeplearning/dpathak/video-processing/epicflow/epicflow'
    # subprocess.call([
    #     command, impath1, impath2, edgefile, deepmatchfile, flofile])
    #
    # # convert flow output binary to txt
    # # flow: (h, w, 2) of format (u=Delta_x, v=Delta_y)
    # flow = read_flo(flofile)
    # using farneback optical flow
    import cv2
    from skimage import color
    im1 = np.array(Image.open(impath1)).astype(float)
    im2 = np.array(Image.open(impath2)).astype(float)
    flow = cv2.calcOpticalFlowFarneback(
        color.rgb2gray(im1), color.rgb2gray(im2), 0.5, 4, 15,
        10, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    # allocate memory: (h * w, 4)
    imH, imW = flow.shape[:2]
    matches = np.zeros((imH * imW, 4), dtype=int)
    # assign (y0, x0)
    matches[:, :2] = np.mgrid[0:imH, 0:imW].T.reshape((-1, 2))
    # (y1, x1) = (v, u) + (y0, x0)
    matches[:, 2:] = flow[matches[:, 0], matches[:, 1]][:, ::-1] + \
        matches[:, :2]
    # (y0, x0, y1, x1) -> (x0, y0, x1, y1)
    # matches maybe out of image size; tobe tructated after per-frame homography
    matches = matches[:, [1, 0, 3, 2]].astype(int)
    # save flitered fg matches to txt file
    # if no homography, then they are already matched (b/c all points)
    np.savetxt(flowMatchFile, matches, fmt='%d')

    if vizFlow:
        import cv2
        hsv = np.zeros_like(im1).astype(np.uint8)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0].astype(float),
                                    flow[..., 1].astype(float))
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 0] = ang * 180 / np.pi / 2
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        imName = flowdir + '/viz_flow/flow_%04d.png' % fNo
        cv2.imwrite(imName, bgr)
        imName = flowdir + '/viz_flow/edge_%04d.png' % fNo
        # import matplotlib.pyplot as plt
        # plt.imsave(imName, (M * 255.).astype(np.uint8))


def run_epicFlow_sequence(imPathList, flowdir, deepmatchdir, vizFlow=False):
    """
    Run EpicFlow Code on a sequence of images of video to obtain tracks.
    To be run after epic flow code.
    """
    fList = []
    if os.path.isdir(flowdir):
        fList = utils.read_r(flowdir, '*.flo')
    if not len(fList) == len(imPathList) - 1 or len(fList) == 0:
        utils.mkdir_p(flowdir)
        if vizFlow:
            utils.mkdir_p(flowdir + '/viz_flow/')
        for i in range(len(imPathList) - 1):
            deepmatchfile = deepmatchdir + 'match_%04d.txt' % i
            run_epicFlow_pair(
                imPathList[i], imPathList[i + 1], flowdir, deepmatchfile, i,
                vizFlow)
            sys.stdout.write('Pairwise EpicFlow: [% 5.1f%%]\r' %
                                (100.0 * float(i / len(imPathList))))
            sys.stdout.flush()
        if vizFlow:
            import subprocess
            subprocess.call(['tar', '-zcf', flowdir + '/viz_flow.tar.gz', '-C',
                                flowdir + '/viz_flow', '.'])
        print('Pairwise EpicFlow completed.')
    else:
        print('Pairwise EpicFlows already present in outdir. Using them.')


def read_dmOutput(fName, imH, imW, scoreTh=-1, fitToIm=True):
    """
    Function for reading output matching files of deepmatch or epicflow
    """
    with open(fName, 'r') as f:
        lines = f.readlines()
    lines = [x.strip().split(' ') for x in lines]
    vals = np.array([[float(y) for y in x] for x in lines])
    if vals.size == 0:
        print('Warning: Empty match file !')
        return vals
    vals[:, :4] = (vals[:, :4] + 0.5).astype(np.int)
    if fitToIm:
        vals[:, :4] = np.minimum(vals[:, :4],
                                    np.array([imW, imH, imW, imH]) - 1)
        vals[:, :4] = np.maximum(vals[:, :4], np.array([0]))
    if scoreTh >= 0:
        vals = vals[vals[:, 4] > scoreTh, :]
    return vals[:, :4]


def match_sequence(imH, imW, matchdir, dmThresh=0, matchNbr=10, shotFrac=0):
    """
    Perform matches for a sequence of pairwise deepmatch
    Input:
        imH, imW: video height and width
        matchdir: dir of matches: x0, y0, x1, y1, score, index
        dmThresh: l-bound for score of deepmatched points to use
        matchNbr: Neighborhood pixel tolerance for tracked points b/w frames
        shotFrac: l-bound on fraction of points to be tracked in a temporal shot
    Output: nFrames x nPoints x 2 : (x,y) of fully tracked points
    """

    def calc_match(list1, list2):
        """
        Input:
            list1: nxd array
            list2: mxd array
        Output:
            matchInd: nx1 array containing indicex of closest pt in list2
            matchDist: nx1 array containing distance from closest pt in list2
        """
        pairwiseDist = np.sqrt(
            np.sum(np.square(list1[..., None] - list2[..., None].T), axis=1))
        matchInd = np.argmin(pairwiseDist, axis=1)
        matchDist = np.amin(pairwiseDist, axis=1)
        return matchInd, matchDist

    matchNbr *= np.sqrt(2)
    mList = utils.read_r(matchdir, '*.txt')

    # total shots: (numFrames,numPts,4)
    # 4 denotes (x,y) of common frame across pairs; kept for accurate homography
    totalShotTracks = np.ones((0, 0, 4), dtype='int')
    currShotTracks = np.ones((0, 0, 4), dtype='int')
    shotends = []  # value is zero indexed and included

    # perform frame wise matching
    for i in range(len(mList)):
        trackCurr = read_dmOutput(mList[i], imH, imW, dmThresh)

        if currShotTracks.size == 0:
            # empty match file: may happen after homography
            if trackCurr.size == 0:
                continue
            initNumTracks = trackCurr.shape[0]
            currShotTracks = np.ones(
                (1 + len(mList), initNumTracks, 4), dtype='int') * -1000
            currShotTracks[i, :, :2] = trackCurr[:, :2]
            currShotTracks[i + 1, :, :2] = trackCurr[:, 2:4]
            continue

        if trackCurr.size != 0:
            matchInd, matchDist = calc_match(
                currShotTracks[i, :, :2], trackCurr[:, :2])
            currNumTracks = matchInd[matchDist <= matchNbr].shape[0]

        if trackCurr.size == 0 or currNumTracks <= shotFrac * initNumTracks:
            # store just finished shot
            shotends.append(i)
            currShotTracks = currShotTracks[:, currShotTracks[i, :, 0] > -1000]
            oldTotal = totalShotTracks
            totalShotTracks = np.ones(
                (1 + i, max(oldTotal.shape[1], currShotTracks.shape[1]), 4),
                dtype='int') * -1000
            totalShotTracks[:oldTotal.shape[0], :oldTotal.shape[1]] = \
                np.copy(oldTotal)
            totalShotTracks[oldTotal.shape[0]:, :currShotTracks.shape[1]] = \
                np.copy(currShotTracks[oldTotal.shape[0]:1 + i])
            currShotTracks = np.ones((0, 0, 4), dtype='int')
        else:
            currShotTracks[i, matchDist <= matchNbr, 2:] = \
                trackCurr[matchInd[matchDist <= matchNbr], :2]
            currShotTracks[i + 1, matchDist <= matchNbr, :2] = \
                trackCurr[matchInd[matchDist <= matchNbr], 2:4]

        sys.stdout.write('Matching sequences for tracking: [% 5.1f%%]\r' %
                            (100.0 * float(i / len(mList))))
        sys.stdout.flush()

    # store last completed shot
    shotends.append(len(mList))
    oldTotal = totalShotTracks
    if currShotTracks.size != 0:
        currShotTracks = currShotTracks[:, currShotTracks[-1, :, 0] > -1000]
        totalShotTracks = np.ones(
            (1 + len(mList), max(oldTotal.shape[1], currShotTracks.shape[1]),
                4), dtype='int') * -1000
        totalShotTracks[:oldTotal.shape[0], :oldTotal.shape[1]] = \
            np.copy(oldTotal)
        totalShotTracks[oldTotal.shape[0]:, :currShotTracks.shape[1]] = \
            np.copy(currShotTracks[oldTotal.shape[0]:])
    else:
        totalShotTracks = np.ones(
            (1 + len(mList), max(oldTotal.shape[1], trackCurr.shape[0]), 4),
            dtype='int') * -1000
        totalShotTracks[:oldTotal.shape[0], :oldTotal.shape[1]] = \
            np.copy(oldTotal)

    print('Matching sequences for tracking completed: %d shots.' %
            len(shotends))
    return totalShotTracks, np.array(shotends)


def run_pre_homography(outdir, matchdir, homTh, imH, imW, dmThresh,
                        imPathList, vizHomo):
    """
    Run per frame homogrpahy on matching files of deepmatch or epic flow
    """
    if homTh < 0:
        return
    utils.rmdir_f(outdir)
    utils.mkdir_p(outdir)
    mList = utils.read_r(matchdir, '*.txt')
    if vizHomo:
        col = np.array([255, 0, 0], dtype='int')
        utils.mkdir_p(outdir + '/viz_homo/')
    for i in range(len(mList)):
        matches = read_dmOutput(mList[i], imH, imW, dmThresh, False)
        matches = frame_homography(matches, homTh)
        # fit to coordinates to image size
        matches = np.minimum(matches, np.array([imW, imH, imW, imH]) - 1)
        matches = np.maximum(matches, np.array([0]))
        matchfile = outdir + 'match_%04d.txt' % i
        np.savetxt(matchfile, matches, fmt='%d')
        if matches.size > 0 and vizHomo:
            im = np.array(Image.open(imPathList[i]))
            im = Image.fromarray(utils.draw_point_im(
                im, matches[:, [1, 0]], col, sizeOut=10))
            im.save(outdir + '/viz_homo/%s' % (imPathList[i].split('/')[-1]))

        sys.stdout.write('Pairwise pre-tracking homogrpahy: [% 5.1f%%]\r' %
                            (100.0 * float(i / len(mList))))
        sys.stdout.flush()

    import subprocess
    subprocess.call(['tar', '-zcf', outdir + '/viz_homo.tar.gz', '-C',
                        outdir + '/viz_homo', '.'])
    print('Pairwise pre-tracking homogrpahy completed.')


def run_dm_pair(impath1, impath2, matchfile):
    """
    Run DeepMatch code between two images
    """
    # make blocking subprocess call to deepmatch command
    # output format: x0, y0, x1, y1, score, index
    import subprocess
    command = os.getenv("HOME") + '/fbcode/_bin/experimental/' + \
        'deeplearning/dpathak/video-processing/deepmatch/deepmatch'
    subprocess.call([
        command, impath1, impath2, '-out', matchfile, '-downscale', '2'])


def run_dm_sequence(outdir, imPathList, frameGap=0, dmThresh=0, matchNbr=10,
                    shotFrac=0, postTrackHomTh=-1, preTrackHomTh=-1,
                    use_epic=False, vizFlow=False, vizTr=False, cpysrc=False):
    """
    Run DeepMatch Code on a sequence of images of video to obtain tracks
    """
    print('Outdir: ', outdir)
    # adjust image list according to frame Gap
    imPathList = imPathList[0:len(imPathList):frameGap + 1]

    # compute pariwise deepmatch across frames
    deepmatchdir = outdir + '/matches/'
    mList = []
    if os.path.isdir(deepmatchdir):
        mList = utils.read_r(deepmatchdir, '*.txt')
    if not len(mList) == len(imPathList) - 1 or len(mList) == 0:
        utils.mkdir_p(deepmatchdir)
        for i in range(len(imPathList) - 1):
            matchfile = deepmatchdir + 'match_%04d.txt' % i
            run_dm_pair(imPathList[i], imPathList[i + 1], matchfile)
            sys.stdout.write('Pairwise DeepMatch: [% 5.1f%%]\r' %
                                (100.0 * float(i / len(imPathList))))
            sys.stdout.flush()
        print('Pairwise DeepMatch completed.')
    else:
        print('Pairwise DeepMatches already present in outdir. Using them.')

    # use epic flow densification process
    if use_epic:
        # TODO: rescore deepmatch
        flowdir = outdir + '/flow/'
        run_epicFlow_sequence(imPathList, flowdir, deepmatchdir, vizFlow)
        matchdir = flowdir
        dmThresh = -1  # deepmatch score no longer matters
    else:
        matchdir = deepmatchdir

    # run homography before matching sequences
    imW, imH = Image.open(imPathList[0]).size
    if preTrackHomTh > 0:
        preHomMatchdir = outdir + '/pre_homographies/'
        run_pre_homography(preHomMatchdir, matchdir, preTrackHomTh,
                            imH, imW, dmThresh, imPathList, True)
        matchdir = preHomMatchdir
        dmThresh = -1  # deepmatch score no longer matters

    # resolve pairwise deep-matches to obtain sequence tracks
    totalShotTracks, shotends = match_sequence(
        imH, imW, matchdir, dmThresh, matchNbr, shotFrac)

    # after above tracking, find foreground points using homography
    if postTrackHomTh > 0:
        startF = 0
        for endF in np.nditer(shotends):
            currshotTracks = totalShotTracks[
                startF:endF + 1, totalShotTracks[endF, :, 0] > -1000]
            fgPts = shot_homography(currshotTracks, postTrackHomTh)
            totalShotTracks[startF:endF + 1, :] = -1000
            totalShotTracks[startF:endF + 1, :fgPts.shape[1]] = fgPts
            startF = endF + 1

    # save matches: no longer need duplicated frame tuples
    totalShotTracks = totalShotTracks[:, :, :2]
    np.save(outdir + '/totalShotTracks.npy', totalShotTracks)
    np.save(outdir + '/shotends.npy', shotends)

    # visualize deepmatch tracks on images and save them
    if vizTr and totalShotTracks.size > 0:
        col = np.array([255, 0, 0], dtype='int')
        totalShotTracks.transpose()
        shotNum = 0
        utils.rmdir_f(outdir + '/viz_tracks/')
        utils.mkdir_p(outdir + '/viz_tracks/%d' % shotNum)
        for i in range(len(imPathList)):
            validPts = totalShotTracks[i, totalShotTracks[i, :, 0] > -1000]
            im = np.array(Image.open(imPathList[i]))
            im = Image.fromarray(utils.draw_point_im(
                im, validPts[:, ::-1], col, sizeOut=10))
            im.save(outdir + '/viz_tracks/%d/%s' %
                        (shotNum, imPathList[i].split('/')[-1]))
            if i == shotends[shotNum] and i < len(imPathList) - 1:
                shotNum += 1
                utils.mkdir_p(outdir + '/viz_tracks/%d' % shotNum)
        import subprocess
        subprocess.call(['tar', '-zcf', outdir + '/viz_tracks.tar.gz', '-C',
                            outdir + '/viz_tracks', '.'])
        print('Track visualization saved.')

    # copy src images to output dir for which tracking performed
    if cpysrc:
        from shutil import copy
        utils.mkdir_p(outdir + '/origImages/')
        for i in range(len(imPathList)):
            copy(imPathList[i], outdir + '/origImages/')
        print('Source images copied to outdir.')


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
        '-dmThresh', dest='dmThresh',
        help='Lowerbound for score of deepmatched points to use',
        default=0, type=float)
    parser.add_argument(
        '-matchNbr', dest='matchNbr',
        help='Neighborhood pixel tolerance for tracked points between frames',
        default=10, type=float)
    parser.add_argument(
        '-shotFrac', dest='shotFrac',
        help='Lower bound on frac of points to be tracked in a temporal shot',
        default=0, type=float)
    parser.add_argument(
        '-postTrackHomTh', dest='postTrackHomTh',
        help='Fg Homography threshold (higher the value, less fg pts).' +
        'Disabled by default. Good Value = 50.',
        default=-1, type=float)
    parser.add_argument(
        '-preTrackHomTh', dest='preTrackHomTh',
        help='Fg Homography threshold (higher the value, less fg pts).' +
        'Disabled by default. Must use with epic_flow. Good Value = 50.',
        default=-1, type=float)
    parser.add_argument(
        '-epic', dest='use_epic', action='store_true',
        help='Use epic flow to densify deepmatch.')
    parser.add_argument(
        '-vizFlow', dest='vizFlow', action='store_true',
        help='Visualize (i.e. save) output epic flow on images.')
    parser.add_argument(
        '-vizTr', dest='vizTr', action='store_true',
        help='Visualize (i.e. save) output tracks on images.')
    parser.add_argument(
        '-cpysrc', dest='cpysrc', action='store_true',
        help='Copy images in video directory to outdir.')
    parser.add_argument(
        '-seed', dest='seed',
        help='Random seed for numpy and python.', default=222, type=int)

    args = parser.parse_args()
    return args


def demo_imagenet():
    """
    Demo of dm_tracker on imagenet videos
    """
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

    # read image files
    imPathList = utils.read_r(args.imdir, '*.*')
    if len(imPathList) < 2:
        print('Not enough images in image directory: \n%s' % args.imdir)
        return

    # run deep match tracker
    run_dm_sequence(args.outdir, imPathList, args.frameGap, args.dmThresh,
                    args.matchNbr, args.shotFrac, args.postTrackHomTh,
                    args.preTrackHomTh, args.use_epic,
                    args.vizFlow, args.vizTr, args.cpysrc)


def demo_videos():
    """
    Demo of dm_tracker on imagenet videos
    """
    args = parse_args()
    np.random.seed(args.seed)

    vidDir = '/home/dpathak/local/data/trash/videos'
    imDir = vidDir + '_im'
    vidPathList = utils.read_r(vidDir, '*.mp4')
    # vidPathList = vidPathList[5:]
    utils.mkdir_p(imDir)
    for i in range(len(vidPathList)):
        print('Video: ', vidPathList[i])
        tmpDir = imDir + '/' + vidPathList[i].split('/')[-1][:-4]
        utils.mkdir_p(tmpDir)
        # imSeq = utils.vid2im(vidPathList[i])
        # assert imSeq.size > 0, "Error reading video file"
        # for j in range(imSeq.shape[0]):
        #     Image.fromarray(imSeq[j]).save(tmpDir + '/frame_%05d.jpg' % j)
        imPathList = utils.read_r(tmpDir, '*.jpg')
        if len(imPathList) < 2:
            print('Not enough images in image directory: \n%s' % tmpDir)
            print('Continuing to next one ...')
            continue
        outdir = tmpDir.split('/')
        outdir[-2] = outdir[-2][:-3] + '_result'
        outdir = '/'.join(outdir)
        run_dm_sequence(outdir, imPathList, args.frameGap, args.dmThresh,
                        args.matchNbr, args.shotFrac, args.postTrackHomTh,
                        args.preTrackHomTh, args.use_epic,
                        args.vizFlow, args.vizTr, args.cpysrc)


if __name__ == "__main__":
    # demo_imagenet()
    demo_videos()
