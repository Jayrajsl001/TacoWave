
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
  author  = {Lakkad, Jayraj S. and Tiwari, Satyam R. and Savaliya, Laksh J. and Prajapati, Akshar V.},
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
TacoWave is released under the **Creative Commons Attribution 4.0 (CC BY 4.0)** license.

