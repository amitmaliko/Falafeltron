# FalafelTron: Transformer-Based Text-to-Speech (TTS) Model

This project was developed as part of a Deep Learning course. Our team first implemented the original Tacotron architecture from scratch. Building upon this implementation, we then developed FalafelTron as an improvement task, enhancing Tacotron with modern transformer-based techniques while addressing several limitations of the original model.

FalafelTron is an improved implementation of Tacotron, maintaining its sequence-to-sequence approach but replacing its RNN-based components with Transformer-based self-attention mechanisms for more efficient, higher-quality speech synthesis.

## Contributors
- Amit Malik
- Nir Manor
- Arbel Askayo

## Background & Motivation

### The Original Tacotron
Tacotron, introduced by Google in 2017, revolutionized text-to-speech synthesis by presenting an end-to-end approach that significantly streamlined traditional TTS pipelines. Unlike previous systems that required separate modules for linguistic feature extraction and duration prediction, Tacotron mapped character inputs directly to mel spectrograms through a sequence-to-sequence model with attention. Tacotron relied on a separate vocoder component (Griffin-Lim algorithm) to convert these spectrograms to actual audio waveforms.

Despite its groundbreaking approach, Tacotron 1 faced several limitations:
- **Sequential Processing**: The RNN-based encoder and decoder limited training parallelization
- **Attention Challenges**: Simple location-sensitive attention struggled with longer sequences
- **Complex Architecture**: The CBHG module (1D convolution banks + highway networks + bidirectional GRU) added unnecessary complexity
- **Vocoder Limitations**: The Griffin-Lim algorithm produced robotic, low-quality speech

### Our Approach: FalafelTron
As the original Tacotron authors embraced tacos, we at Haifa University drew inspiration from our local favorite - the falafel. Thus, FalafelTron was born, with the mission to address Tacotron's limitations while maintaining its elegant end-to-end approach.

Our key innovations include:
- **Transformer Architecture**: Replacing RNNs with self-attention mechanisms for better parallelization
- **Simplified Design**: Eliminating the CBHG module in favor of a cleaner Transformer-based structure
- **Enhanced Attention**: Using multi-head self-attention for improved text-audio alignment
- **WaveGlow Integration**: Implementing NVIDIA's flow-based vocoder for higher-quality audio synthesis

This project represents an educational exploration into modern TTS architectures, demonstrating how Transformer-based approaches can enhance speech synthesis while providing valuable insights into deep learning model design and optimization.

## Overview

This project enhances the original Tacotron 1 model using Transformer architectures, resulting in:
- Faster training through improved parallelization
- Better attention learning and alignment
- Higher quality speech synthesis
- Maintenance of core sequence-to-sequence principles

## Architecture

FalafelTron follows a sequence-to-sequence approach:

**Text Input → Character Embeddings → Transformer Encoder → Transformer Decoder → Mel Spectrogram Output → WaveGlow Vocoder → Final Speech Audio**

### Key Components

1. **Positional Encoding**
   - Injects sequence order information
   - Ensures words are processed with correct timing

2. **Transformer Encoder**
   - Processes text embeddings with multi-head self-attention
   - Captures phoneme dependencies without recurrence

3. **Transformer Decoder**
   - Uses self-attention to model temporal speech dependencies
   - Cross-attention aligns text with predicted mel spectrogram frames
   - PreNet processes previous mel frames before decoding

4. **Custom Transformer Decoder Layer**
   - Stores attention weights for analysis and debugging
   - Tracks text-spectrogram alignments

5. **WaveGlow Vocoder**
   - Converts mel spectrograms to high-quality audio
   - Eliminates need for iterative Griffin-Lim processing

## Improvements Over Tacotron 1

| Feature | Tacotron 1 | FalafelTron |
|---------|------------|-------------|
| Architecture | RNN-based | Transformer-based |
| Processing | Sequential | Parallel |
| Attention | Location-sensitive | Self-attention |
| Training Speed | Slower | 2-4x faster |
| Vocoder | Griffin-Lim | WaveGlow |
| Feature Engineering | CBHG & PostNet | Simplified architecture |

## Training Details

### Dataset
- LJ Speech Dataset (13,100 short audio clips from a single speaker)
- 80% Training / 20% Validation split
- Optimized DataLoader with dynamic padding and multi-threading

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 16 |
| Learning Rate | 0.0001 |
| d_model | 256 |
| Attention Heads | 4 |
| Encoder Layers | 4 |
| Decoder Layers | 4 |
| Reduction Factor | 2 |
| Mel Spectrogram Dim | 80 |
| Optimizer | AdamW |

- Cosine Annealing LR Decay
- Teacher forcing applied in early training

## Results & Audio Samples

Audio samples generated with FalafelTron can be found in the `wavs` directory, comparing different training durations:
- FalafelTron15epochs.wav
- FalafelTron20epochs.wav
- Tacotron_Output.wav (for comparison)

## Getting Started

### Prerequisites

```
torch>=1.8.0
torchaudio
librosa
numpy
matplotlib
pandas
tqdm
scipy
soundfile
wandb (for experiment tracking)
```

### Installation

1. Clone this repository:
```bash
git clone https://github.com/amitmaliko/falafeltron.git
cd falafeltron
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the LJSpeech dataset (if training from scratch)

### Usage

See the `Falafeltron.ipynb` notebook for a complete walkthrough of:
- Data preparation
- Model training
- Inference
- Audio generation

## Future Improvements

- Replace L1 Loss with Spectral Loss for improved perceptual quality
- Explore different learning rate schedules
- Experiment with hybrid attention mechanisms
- Train longer on larger datasets
- Optimize stop-prediction further
- Compare different vocoders (HiFi-GAN, WaveRNN)

## Acknowledgments

- Google's Tacotron research team for the original paper:
  - Wang, Y., Skerry-Ryan, R.J., Stanton, D., Wu, Y., Weiss, R.J., Jaitly, N., Yang, Z., Xiao, Y., Chen, Z., Bengio, S., Le, Q., Agiomyrgiannakis, Y., Clark, R., & Saurous, R.A. (2017). Tacotron: Towards End-to-End Speech Synthesis. arXiv:1703.10135.

- NVIDIA for the WaveGlow vocoder:
  - Prenger, R., Valle, R., & Catanzaro, B. (2019). WaveGlow: A Flow-based Generative Network for Speech Synthesis. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 3617-3621). IEEE.

- LJSpeech dataset creators

## License

This project is licensed under the MIT License

## Citation

If you use this work in your research, please cite:

```
@misc{falafeltron2025,
  author = {Malik, Amit and Manor, Nir and Askayo, Arbel},
  title = {FalafelTron: Transformer-Based Text-to-Speech Model},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/amitmaliko/falafeltron}}
}
``` 