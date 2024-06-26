## Multimodal Active Measurement for Human Mesh Recovery in Close Proximity

Code repository for the paper:

**Multimodal Active Measurement for Human Mesh Recovery in Close Proximity**

Takahiro Maeda, Keisuke Takeshita, Norimichi Ukita, and Kazuhito Tanaka

Work done during the internship at Frontier Research Center, Toyota Motor Corporation

[[paper](https://arxiv.org/abs/2310.08116)]

![teaser](teaser.png)


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

We provide the preprocessed annotation of 3DPW-OC [here](https://drive.google.com/drive/folders/1ZoIf9k3fjkdvW-KjooB1xYdT2IoNMZLh?usp=drive_link). You must merge the downloaded dir to `data/`.

## Evaluation
```
python eval.py
```
