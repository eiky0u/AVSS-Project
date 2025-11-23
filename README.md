<h1 align="center">
  Audio-Visual Source Separation with PyTorch
</h1>

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#references">References</a> •
  <a href="#credits">Credits</a>
</p>

## About

This repository implements an **audio-visual speaker separation (AVSS)** system on top of the PyTorch Project Template.  
The goal is to separate individual speakers from a single-channel audio mixture by leveraging both the **audio track** and **visual cues** from the speakers’ mouth regions.

Our implementations follow the original audio-visual separation architectures proposed in [TDFNet](https://arxiv.org/abs/2401.14185) and [RTFSNet](https://arxiv.org/abs/2309.17189).

Both models consume:

- an audio mixture (`mix`)  
- video tensors with mouth regions (`mouths`)

but differ in how they encode and fuse visual features with the audio pathway.

We evaluate the models using **SI-SNRi** as the main separation metric.  
In our experiments:

- **TDFNet**  reaches **10.253 SI-SNRi**, **10.896 SDR**, **1.873 PESQ**, **0.877 STOI**;
- **RTFSNet** reaches **8.258 SI-SNRi**, **9.041 SDR**, **1.723 PESQ**, **0.844 STOI**.

The rest of the repository provides configuration, training, inference, waveform saving, and a one-batch speed & resource benchmark for these AVSS architectures.


## Installation

Follow these steps to install the project:

1. Install all required packages:

   ```bash
   pip install -r requirements.txt
   ```
2. Download pretrained model weights from Hugging Face:
   ```bash
   git clone https://huggingface.co/eikyou/TDFNet
   git clone https://huggingface.co/Dodenus/RTFSNet
   ```

## How To Use

To train a model, run:

```bash
python train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments, such as batch size, path to data folders, etc.

To run inference (and save predictions):

```bash
python inference.py model=MODEL_NAME inferencer.from_pretrained=PATH_TO_WEIGHTS HYDRA_CONFIG_ARGUMENTS
```
where `MODEL_NAME` is one of the `tdfnet_22` `tdfnet_16` `rtfsnet` and `PATH_TO_WEIGHTS` is the path to the weights of the corresponding model.

To run the evaluation script:
```bash
python calc_metrics.py --gt-root PATH_TO_GT --pred-root PATH_TO_PRED --sr SR
```
where `PATH_TO_GT` is the path to the ground truth data, `PATH_TO_PRED` is the path to the predicted data and `SR` is the target sample rate.

To run the speed benchmark script:
```bash
python inference.py \
  +inferencer.speed_test=true \
  +datasets.inference.limit=1 \
  dataloader.batch_size=1 \
  model=MODEL_NAME \
  inferencer.from_pretrained=PATH_TO_WEIGHTS
```
where `MODEL_NAME` is one of the `tdfnet_22` `tdfnet_16` `rtfsnet` and `PATH_TO_WEIGHTS` is the path to the weights of the corresponding model.

## References

- TDFNet: An Efficient Audio-Visual Speech Separation Model with Top-down Fusion – [paper](https://arxiv.org/abs/2401.14185)
- RTFS-Net: Recurrent Time-Frequency Modelling for Efficient Audio-Visual Speech Separation – [paper](https://arxiv.org/abs/2309.17189)
- Lip-reading backbone: [Lipreading_using_Temporal_Convolutional_Networks](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks)


## Credits

This repository is based on the [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).
