## Video Segmentation and tracking

Code for unsupervised bottom-up video motion segmentation. uNLC is a reimplementation of the NLC algorithm by Faktor and Irani, BMVC 2014, that removes the trained edge detector and makes numerous other modifications and simplifications. For additional details, see section 5.1 in the <a href="http://cs.berkeley.edu/~pathak/unsupervised_video/">paper</a>. This repository also contains code for a very simple video tracker which we developed.

This code was developed and is used in our [CVPR 2017 paper on Unsupervised Learning using unlabeled videos](http://cs.berkeley.edu/~pathak/unsupervised_video/). Github repository for our CVPR 17 paper is [here](https://github.com/pathak22/unsupervised-video). If you find this work useful in your research, please cite:

    @inproceedings{pathakCVPR17learning,
        Author = {Pathak, Deepak and Girshick, Ross and Doll\'{a}r,
                  Piotr and Darrell, Trevor and Hariharan, Bharath},
        Title = {Learning Features by Watching Objects Move},
        Booktitle = {Computer Vision and Pattern Recognition ({CVPR})},
        Year = {2017}
    }
<hr/>


**Video Segmentation** using low-level vision based unsupervised methods. It is largely inspired from Non-Local Consensus [Faktor and Irani, BMVC 2014] method, but removes all trained components. This segmentation method includes and make use of code for optical flow, motion saliency code, appearance saliency, superpixel and low-level descriptors.

**Video Tracking** code includes deepmatch followed by epic flow (or farneback) and then doing homography followed by bipartite matching to obtain foreground tracks.

### Setup

1. Install optical flow
  ```Shell
  cd videoseg/lib/
  git clone https://github.com/pathak22/pyflow.git
  cd pyflow/
  python setup.py build_ext -i
  python demo.py    # -viz option to visualize output
  ```

2. Install Dense CRF code
  ```Shell
  cd videoseg/lib/
  git clone https://github.com/lucasb-eyer/pydensecrf.git
  cd pydensecrf/
  python setup.py build_ext -i
  PYTHONPATH=.:$PYTHONPATH python examples/inference.py examples/im1.png examples/anno1.png examples/out_new1.png
  # compare out_new1.png and out1.png -- they should be same
  ```

3. Install appearance saliency
  ```Shell
  cd videoseg/lib/
  git clone https://github.com/ruanxiang/mr_saliency.git
  ```

4. Install kernel temporal segmentation code
  ```Shell
  # cd videoseg/lib/
  # wget http://pascal.inrialpes.fr/data2/potapov/med_summaries/kts_ver1.1.tar.gz
  # tar -zxvf kts_ver1.1.tar.gz && mv kts_ver1.1 kts
  # rm -f kts_ver1.1.tar.gz

  # Edit kts/cpd_nonlin.py to remove weave dependecy. Due to this change, we are shipping the library.
  # Included in videoseg/lib/kts/ . However, it is not a required change if you already have weave installed
  # (which is mostly present by default).
  ```

5. Convert them to modules
  ```Shell
  cd videoseg/lib/
  cp __init__.py mr_saliency/
  cp __init__.py kts/
  ```

6. Run temporal segmentation:
  ```Shell
  time python vid2shots.py -imdir /home/dpathak/local/data/trash/my_nlc/imseq/v21/ -out /home/dpathak/local/data/trash/my_nlc/nlc_out/
  ```

7. Run NLC segmentation:
  ```Shell
  cd videoseg/src/
  time python nlc.py -imdir /home/dpathak/local/data/trash/my_nlc/imseq/3_tmp/ -out /home/dpathak/local/data/trash/my_nlc/nlc_out/ -maxsp 400 -iters 100
  ```

8. Run Tracker:
  ```Shell
  cd videoseg/src/
  time python dm_tracker.py -fgap 2 -seed 2905 -vizTr -dmThresh 0 -shotFrac 0.2 -matchNbr 20 -postTrackHomTh -1 -preTrackHomTh 10
  ```

9. Run CRF sample:
  ```Shell
  cd videoseg/src/
  time python crf.py -inIm ../lib/pydensecrf/examples/im1.png -inL ../lib/pydensecrf/examples/anno1.png -out ../lib/pydensecrf/examples/out_new2.png
  ```

10. Run Full Pipeline:
  ```Shell
  cd videoseg/src/
  time python run_full.py -out /home/dpathak/local/data/AllSegInit -in /home/dpathak/fbcode/experimental/deeplearning/dpathak/videoseg/datasets/imdir_Samples.txt
  ```
