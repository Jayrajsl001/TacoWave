# Tacotron 2 (without wavenet)

PyTorch implementation of [Natural TTS Synthesis By Conditioning
Wavenet On Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf). 

This implementation includes **distributed** and **automatic mixed precision** support
and uses the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/).

Distributed and Automatic Mixed Precision support relies on NVIDIA's [Apex] and [AMP].

Visit our [website] for audio samples using our published [Tacotron 2] and
[WaveGlow] models.

![Alignment, Predicted Mel Spectrogram, Target Mel Spectrogram](tensorboard.png)


## Pre-requisites
1. NVIDIA GPU + CUDA cuDNN

## Setup
1. Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/)
2. Clone this repo: `git clone https://github.com/NVIDIA/tacotron2.git`
3. CD into this repo: `cd tacotron2`
4. Initialize submodule: `git submodule init; git submodule update`
5. Update .wav paths: `sed -i -- 's,DUMMY,ljs_dataset_folder/wavs,g' filelists/*.txt`
    - Alternatively, set `load_mel_from_disk=True` in `hparams.py` and update mel-spectrogram paths 
6. Install [PyTorch 1.0]
7. Install [Apex]
8. Install python requirements or build docker image 
    - Install python requirements: `pip install -r requirements.txt`

## Training
1. `python train.py --output_directory=outdir --log_directory=logdir`
2. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## Training using a pre-trained model
Training using a pre-trained model can lead to faster convergence  
By default, the dataset dependent text embedding layers are [ignored]

1. Download our published [Tacotron 2] model
2. `python train.py --output_directory=outdir --log_directory=logdir -c tacotron2_statedict.pt --warm_start`

## Multi-GPU (distributed) and Automatic Mixed Precision Training
1. `python -m multiproc train.py --output_directory=outdir --log_directory=logdir --hparams=distributed_run=True,fp16_run=True`

## Inference demo
1. Download our published [Tacotron 2] model
2. Download our published [WaveGlow] model
3. `jupyter notebook --ip=127.0.0.1 --port=31337`
4. Load inference.ipynb 

N.b.  When performing Mel-Spectrogram to Audio synthesis, make sure Tacotron 2
and the Mel decoder were trained on the same mel-spectrogram representation. 


## Related repos
[WaveGlow](https://github.com/NVIDIA/WaveGlow) Faster than real time Flow-based
Generative Network for Speech Synthesis

[nv-wavenet](https://github.com/NVIDIA/nv-wavenet/) Faster than real time
WaveNet.

## Acknowledgements
This implementation uses code from the following repos: [Keith
Ito](https://github.com/keithito/tacotron/), [Prem
Seetharaman](https://github.com/pseeth/pytorch-stft) as described in our code.

We are inspired by [Ryuchi Yamamoto's](https://github.com/r9y9/tacotron_pytorch)
Tacotron PyTorch implementation.

We are thankful to the Tacotron 2 paper authors, specially Jonathan Shen, Yuxuan
Wang and Zongheng Yang.


[WaveGlow]: https://drive.google.com/open?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF
[Tacotron 2]: https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view?usp=sharing
[pytorch 1.0]: https://github.com/pytorch/pytorch#installation
[website]: https://nv-adlr.github.io/WaveGlow
[ignored]: https://github.com/NVIDIA/tacotron2/blob/master/hparams.py#L22
[Apex]: https://github.com/nvidia/apex
[AMP]: https://github.com/NVIDIA/apex/tree/master/apex/amp









# TacoWave: Fast and High-Fidelity End-to-End Speech Synthesis

TacoWave is a fast and high-quality **end-to-end (E2E) speech synthesis**
framework designed to generate natural-sounding speech with **low inference
latency** and **high perceptual fidelity**. The model balances the trade-off
between **model size, synthesis accuracy, and real-time performance**, often
referred to as the *magic triangle* in speech synthesis research.

TacoWave extends the attention-based Tacotron family by incorporating
**forward attention**, **GELU activations**, and a **deepened PostNet**, and
leverages **BigVGAN** for high-fidelity waveform generation from Mel
spectrograms.

---

## 🔍 Key Contributions

- Forward attention mechanism for **stable and monotonic alignment**
- GELU activation for improved **prosody preservation and training stability**
- Deepened PostNet for refined Mel-spectrogram quality
- End-to-end integration with **BigVGAN** neural vocoder
- Fast inference with **47.16× real-time factor** on NVIDIA GTX 1080

---

## 🧠 Model Overview

TacoWave follows a two-stage pipeline:

1. **Text → Mel Spectrogram**
   - Character embedding
   - Convolutional encoder with GELU
   - Bidirectional LSTM
   - Forward attention-based decoder
   - Deep PostNet refinement

2. **Mel Spectrogram → Waveform**
   - BigVGAN GAN-based neural vocoder

This design enables high-quality waveform synthesis while maintaining
real-time capability.

---

## 📊 Dataset

Experiments were conducted using the **LJ Speech Dataset**, which contains
high-quality recordings from a single female speaker with aligned transcripts.

- Sampling rate: 22,050 Hz  
- Total duration: ~24 hours  
- Total utterances: 13,100  

---

## 🧰 Pre-requisites

- NVIDIA GPU with CUDA and cuDNN
- Python ≥ 3.8
- PyTorch
- NVIDIA Apex (optional, for mixed-precision training)
- LJSpeech Dataset

---

## ⚙️ Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd tacowave
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
```bash
wget https://keithito.com/LJ-Speech-Dataset/LJSpeech-1.1.tar.bz2
tar -xvf LJSpeech-1.1.tar.bz2
```

### 4. Configure Dataset Paths
Update dataset and wav paths in the configuration or filelists before training.

---

## 🏋️ Training

To train TacoWave from scratch:

```bash
python train.py   --output_directory=outdir   --log_directory=logdir
```

(Optional) Monitor training using TensorBoard:
```bash
tensorboard --logdir=outdir/logdir
```

---

## 🔁 Training with Pretrained Weights

Training can be initialized using pretrained Tacotron-style weights to speed
up convergence:

```bash
python train.py   --output_directory=outdir   --log_directory=logdir   --warm_start
```

---

## ⚡ Distributed & Mixed Precision Training

```bash
python -m multiproc train.py   --output_directory=outdir   --log_directory=logdir   --hparams=distributed_run=True,fp16_run=True
```

---

## 🎧 Inference

- Load a trained TacoWave checkpoint
- Generate Mel spectrograms from text
- Convert Mel spectrograms to waveform using BigVGAN

⚠️ Ensure both TacoWave and BigVGAN use the **same Mel configuration**.

---

## 📈 Evaluation Metrics

### Subjective
- Mean Opinion Score (MOS)
- NISQA-MOS

### Quantitative
- Character Error Rate (CER)
- Word Error Rate (WER)

### Objective
- PESQ
- MCD
- MSD
- STOI
- PCC
- Cosine Similarity (CS)

---

## 🧪 Results Summary

- MOS: **4.21**
- NISQA-MOS: **3.35**
- Inference Speed: **47.16× real-time**
- Improved prosody, intelligibility, and spectral consistency

---

## ⚠️ Limitations

- Forward attention alignment is difficult to debug in long-form synthesis
- Slight degradation for out-of-domain text
- Lack of explicit attention diagnostic tools

---

## 🔮 Future Work

- Attention alignment visualization tools
- Improved robustness for long and unseen text
- Model compression and lightweight variants
- Multi-speaker and multilingual extensions

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@article{lakkad2025tacowave,
  title   = {TacoWave: Fast and High-Fidelity End-to-End Speech Synthesis},
  author  = {Lakkad, Jayraj S. and Tiwari, Satyam R. and Savaliya, Laksh J.},
  year    = {2025}
}
```

---

## 🙏 Acknowledgements

This research is inspired by and builds upon:
- NVIDIA Tacotron 2
- Keith Ito
- Prem Seetharaman
- Ryuichi Yamamoto

We sincerely thank the authors of Tacotron, Tacotron 2, and BigVGAN for their
foundational contributions to speech synthesis research.

---

## 📜 License

This project is intended for **academic and research purposes only**.
