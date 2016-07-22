## Video Segmentation and tracking

**Video Segmentation** using low-level vision based unsupervised methods. It is largely inspired from Non-Local Consensus method, but completely unsupervised. This segmentation method includes and make use of code for optical flow, motion saliency code, appearance saliency, superpixel and low-level descriptors.

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

2. Install appearance saliency
  ```Shell
  cd videoseg/lib/
  git clone https://github.com/ruanxiang/mr_saliency.git
  ```

3. Install kernel temporal segmentation code
  ```Shell
  cd videoseg/lib/
  wget http://pascal.inrialpes.fr/data2/potapov/med_summaries/kts_ver1.1.tar.gz
  tar -zxvf kts_ver1.1.tar.gz && mv kts_ver1.1 kts
  rm -f kts_ver1.1.tar.gz
  ```

4. Convert them to modules
  ```Shell
  cd videoseg/lib/
  cp __init__.py mr_saliency/
  cp __init__.py kts/
  ```

5. Run temporal segmentation:
  ```Shell
  time python vid2shots.py -imdir /home/dpathak/local/data/trash/my_nlc/imseq/v21/ -out /home/dpathak/local/data/trash/my_nlc/nlc_out/
  ```

6. Run NLC segmentation:
  ```Shell
  cd videoseg/src/
  time python nlc.py -imdir /home/dpathak/local/data/trash/my_nlc/imseq/3_tmp/ -out /home/dpathak/local/data/trash/my_nlc/nlc_out/ -maxsp 400 -iters 100 -seed 2905 -fgap 0
  ```

7. Run Tracker:
  ```Shell
  cd videoseg/src/
  time python dm_tracker.py -fgap 2 -seed 2905 -vizTr -dmThresh 0 -shotFrac 0.2 -matchNbr 20 -postTrackHomTh -1 -preTrackHomTh 10
  ```
