## Multimodal Active Measurement for Human Mesh Recovery in Close Proximity

Code repository for the paper:

**Multimodal Active Measurement for Human Mesh Recovery in Close Proximity**

Takahiro Maeda, Keisuke Takeshita, Norimichi Ukita, and Kazuhito Tanaka

Work done during the internship at Frontier Research Center, Toyota Motor Corporation

[[paper](https://arxiv.org/abs/2310.08116)]

![teaser](teaser.png)

## Abstract

For physical human-robot interactions (pHRI), a robot needs to estimate the accurate body pose of a target person.
However, in these pHRI scenarios, the robot cannot fully observe the target person's body with equipped cameras because the target person must be close to the robot for physical interaction.
This closeness leads to severe truncation and occlusions and thus results in poor accuracy of human pose estimation.
For better accuracy in this challenging environment, we propose an **active measurement** and **sensor fusion** framework of the equipped cameras with touch and ranging sensors such as 2D LiDAR.
Touch and ranging sensor measurements are sparse but reliable and informative cues for localizing human body parts.
In our **active measurement** process, camera viewpoints and sensor placements are dynamically optimized to measure body parts with higher estimation uncertainty, which is closely related to truncation or occlusion.
In our **sensor fusion** process, assuming that the measurements of touch and ranging sensors are more reliable than the camera-based estimations, we fuse the sensor measurements to the camera-based estimated pose by aligning the estimated pose towards the measured points.
Our proposed method outperformed previous methods on the standard occlusion benchmark with simulated active measurement.
Furthermore, our method reliably estimated human poses using a real robot even with practical constraints such as occlusion by blankets.


## Dependencies

```
git submodule update --init
pip install -r requirements.txt
cd ProHMR
python setup.py developw
```

## Data Preparation

This repository is based on [ProHMR](https://github.com/nkolot/ProHMR/). You need to some components by the script below.
```
bash fetch_data.sh
```
To download the SMPL model, please follow the **Fetch data** instruction in the ProHMR repository.
For the evaluation on the 3DPW-OC dataset, you need to download the images from [official 3DPW webpage](https://virtualhumans.mpi-inf.mpg.de/3DPW/) (only imageFiles).
You must change the `configs/datasets.yaml` accordingly to your 3DPW location.

We provide the preprocessed annotation of 3DPW-OC [here](https://drive.google.com/file/d/1ZnQ8YD50Gx1tC7y0rTBWkJ5HtxaBEfAU/view?usp=drive_link). You must place the downloaded npz at `data/datasets`.

## Evaluation
```
python eval.py
```
