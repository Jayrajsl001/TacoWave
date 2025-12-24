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

TacoWave is a fast, high-quality end-to-end (E2E) speech synthesis model designed to balance the *magic triangle* of **model size, synthesis accuracy, and inference speed**. The model predicts Mel spectrograms from text using an attention-based architecture and synthesizes waveforms using **BigVGAN** as a neural vocoder.

This repository accompanies the research paper:

**TacoWave: Fast and High-Fidelity End-to-End Speech Synthesis Waveform Generation**

---

## 🔍 Overview

Recent E2E speech synthesis models achieve high perceptual quality at the cost of heavy computation and slow inference. TacoWave addresses this limitation by:

- Introducing **forward attention** for stable and monotonic alignment
- Replacing ReLU with **GELU activations** across convolutional components
- Using a **deeper PostNet** for improved Mel-spectrogram refinement
- Integrating **BigVGAN** for high-fidelity waveform generation

TacoWave achieves **47.16× real-time inference** on an NVIDIA GTX 1080 (8GB) while maintaining strong perceptual and intelligibility performance.

---

## 🧠 Model Architecture

TacoWave follows a text-to-Mel-to-waveform pipeline:

### Text-to-Mel Spectrogram
- Character embedding (768-dim)
- 5 × Conv1D layers (kernel size = 7, channels = 768)
- Bidirectional LSTM (256 units per direction)
- Forward attention mechanism
- Decoder with 2 × LSTM layers (hidden size = 1280)
- PostNet with 7 × Conv1D layers + BatchNorm + GELU

### Mel-to-Waveform
- **BigVGAN** neural vocoder
- Anti-aliased Multi-Periodicity (AMP) modules
- Multi-scale and multi-period discriminators

---

## ⚙️ Key Design Choices

- **Forward Attention**  
  Enforces soft monotonic alignment for long-form and stable synthesis.

- **GELU Activation**  
  Replaces ReLU to preserve low-energy phonetic and prosodic cues, improving gradient flow and training stability.

- **Deep PostNet**  
  Enhances Mel-spectrogram refinement and spectral smoothness.

---

## 📊 Dataset

Experiments are conducted using the **LJ Speech Dataset**:

| Property | Value |
|--------|-------|
| Speaker | Single female |
| Sampling Rate | 22,050 Hz |
| Total Clips | 13,100 |
| Total Duration | ~24 hours |
| Avg Clip Duration | 6.57 sec |

---

## 🏋️ Training Setup

- GPU: NVIDIA GTX 1080 (8GB)
- CPU: Intel Core i7 (12th Gen)
- OS: Ubuntu 24.04 LTS
- Batch Size: 16
- Training Steps: up to 50K
- Learning Rate: 1e−3

---

## 📈 Evaluation Metrics

### Subjective Metrics
- **MOS** (Mean Opinion Score)
- **NISQA-MOS**

### Quantitative Metrics
- **CER** (Character Error Rate)
- **WER** (Word Error Rate)

### Objective Metrics
- PESQ
- MCD
- MSD
- STOI
- PCC
- Cosine Similarity (CS)

---

## 🧪 Results Summary

| Model | MOS ↑ | NISQA-MOS ↑ | CER ↓ | WER ↓ |
|------|------|-------------|-------|-------|
| TacoWave V1 | 4.15 | 3.47 | 1.32 | 1.78 |
| **TacoWave V2** | **4.21** | **3.35** | **0.93** | **1.30** |

- Achieves **47.16× real-time inference**
- Strong intelligibility and naturalness
- Better spectral continuity and harmonic richness compared to baseline vocoders

---

## 🚀 Inference Speed (RTF)

| Model | GPU | Real-Time Factor |
|------|-----|------------------|
| TacoWave V2 | GTX 1080 | **47.16×** |
| TacoWave V1 | GTX 1080 | 23.25× |

---

## 🧩 Known Limitations

- Forward attention alignment can be hard to debug in long or out-of-domain text
- Slightly higher WER in rare unseen scenarios
- Lack of diagnostic visualization tools for attention misalignment

---

## 📌 Future Work

- Attention visualization and diagnostic tools
- Robust alignment debugging for long-form synthesis
- Enhanced generalization to out-of-domain text
- Further reduction in model footprint

---

## 📄 Citation

If you use TacoWave in your research, please cite:

```bibtex
@article{lakkad2025tacowave,
  title   = {TacoWave: Fast and High-Fidelity End-to-End Speech Synthesis Waveform Generation},
  author  = {Lakkad, Jayraj S. and Tiwari, Satyam R. and Savaliya, Laksh J.},
  year    = {2025}
}
