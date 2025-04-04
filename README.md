RawNet2 Audio Deepfake Detection
Detect AI-generated (spoofed) human speech using a lightweight version of the RawNet2 model. This repository explores deepfake audio detection by directly processing raw waveforms and analyzing temporal speech patterns using a simplified RawNet2 architecture.

📌 Project Overview
This repository implements a simplified RawNet2 model to detect bonafide vs. spoofed audio from raw waveform inputs. It simulates a realistic audio deepfake detection pipeline using dummy data, with the option to plug in real datasets like ASVspoof 2019 LA.

🔍 Research & Approach Selection
After examining the Audio Deepfake Detection GitHub repository, we selected RawNet2 as our primary architecture. Here's a summary of the research evaluation:

✅ Approach 1: RawNet2 (ICASSP 2020)
End-to-end DNN processing raw audio with residual CNN + GRU.

EER ~1.08% on ASVspoof 2019 — high performance.

GRU captures long-term dependencies across speech sequences.

Tradeoffs: Needs strong GPU support; sensitive to raw audio noise.

🧠 Approach 2: LFCC / CQCC + GMM (Traditional)
Classical ML using cepstral features + Gaussian modeling.

EER ~17.55% — lighter but less accurate.

Advantages: Easy deployment on embedded systems.

Drawbacks: Weak generalization to new spoof types.

🛡️ Approach 3: AASIST (INTERSPEECH 2022)
Multi-stream architecture with Res2Net + attention.

State-of-the-art: EER as low as 0.63%.

Highly modular and robust in noisy settings.

Tradeoffs: Large model, may require pruning for deployment.

⚙️ Implementation Details
Model Architecture
We implemented a lightweight RawNet2-style model using:

Conv1D layers to extract local acoustic patterns.

GRU layer to model sequential speech dynamics.

FC + Sigmoid layer for binary classification (spoof/bonafide).

Dataset
A dummy dataset was created using random waveforms (simulating audio inputs).

Labels alternate between bonafide (0) and spoof (1).

Sample rate: 16kHz, 1-second audio clips.

🛠️ Training & Evaluation
Loss: Binary Cross-Entropy (BCE)

Optimizer: Adam (LR = 0.001)

Epochs: 5

Evaluation: Accuracy + classification_report from sklearn

Note: Since dummy data is used, performance varies randomly and does not reflect real-world model accuracy.

📊 Documentation & Analysis
1. Implementation Process
🚧 Challenges
No real audio deepfake data in early phases.

GRU training instability with dummy input lacking temporal meaning.

🔄 Solutions
Dummy dataset created with binary alternating labels.

Simplified RawNet2Lite version to reduce model size and overfitting risk.

🧠 Assumptions
Audio clips are 1s long, sampled at 16kHz.

Balanced dataset between spoof and bonafide.

Prototype is for demonstrating the feasibility of end-to-end processing.

2. Model Analysis
🔎 Why RawNet2?
End-to-end processing of raw audio without handcrafted features.

Good benchmark results on public ASVspoof datasets.

GRU enables sequential understanding of audio patterns.

🧬 Model Workflow
mathematica
Copy
Edit
Input (1D waveform) → Conv1D Layers → GRU → FC → Sigmoid → Prediction
📈 Performance (on dummy data)
Accuracy fluctuates between 60–70% depending on randomness.

Functional pipeline confirmed, not for performance benchmarking.

3. Strengths & Weaknesses
✅ Strengths
Fully modular & extendable.

Ideal for prototyping raw audio input pipelines.

Ready for integration with APIs (e.g., FastAPI, Flask).

❌ Weaknesses
Dummy data lacks real-world variability.

GRU may struggle with long or noisy audio sequences.

💡 Future Improvements
Integrate real datasets: ASVspoof 2019 LA, WaveFake, TIMIT.

Upgrade to LSTM + attention or Transformers for better modeling.

Use pretrained RawNet2 weights for improved performance.

Add features: post-processing, confidence thresholds, and UI/REST APIs.

🔄 Reflection Questions
Most significant challenge?
Lack of real, labeled spoof audio and balancing simplicity vs. fidelity in RawNet2Lite.

Real-world performance?
Likely lower due to noise, accents, compression — domain adaptation would help.

What would improve it?
Real-world spoofed datasets, more training compute, augmentation techniques.

How to deploy?

TorchScript/ONNX for optimized inference

FastAPI/Flask for serving

Preprocessing with VAD & normalization

Monitoring with human-in-the-loop for edge cases
