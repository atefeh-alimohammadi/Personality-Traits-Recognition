# Personality Traits Recognition from Video via Fusion of Visual Features

This repository contains the code for our BSc thesis project on **apparent personality trait recognition** from facial videos, using a deep learning pipeline that fuses visual features (raw frames and facial landmarks) on the **ChaLearn First Impressions** dataset.

The goal of this work is to estimate continuous **Big Five personality traits** (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) from short video clips where individuals speak to the camera.

---

## 1. Project Overview

In this project, **we adapt a Spatial-to-Dynamic Learning (S2D) architecture combined with a Vision Transformer (ViT) backbone** for regression-based apparent personality prediction.

The model receives two parallel inputs:

- **Raw visual frames**
- **Facial landmarks**

and learns to fuse spatial and temporal information using S2D-inspired modules to produce personality scores in the range \[0, 1].

---

## 2. Relation to Previous Work (2024 Model Inspiration)

This project is inspired by the S2D-based spatial–temporal fusion architecture proposed in the 2024 emotion-recognition paper:

**[From Static to Dynamic:
Adapting Landmark-Aware Image Models
for Facial Expression Recognition in Videos]**  
*Authors: Yin Chen†
, Jia Li†∗, Shiguang Shan, Fellow, IEEE,
Meng Wang, Fellow, IEEE, and Richang Hong, Member, IEEE, 2024*  
[https://arxiv.org/abs/2312.05447]()

While the original work focused on **emotion classification**, we:

- adapted the architecture for **continuous personality regression**,  
- re-designed the regression head for five Big Five traits,  
- fine-tuned MCP and TMA modules to emphasize personality-relevant dynamics,  
- integrated new evaluation metrics, and  
- restructured the training engine for our task.

This project shows how an affective computing architecture can be **repurposed and customized** for personality analysis.

---

## 3. Dataset and Preprocessing

We use the publicly available **ChaLearn First Impressions** dataset, which contains short YouTube-style interview videos annotated with continuous Big Five scores.

### Preprocessing

We did **not** perform low-level preprocessing (frame extraction, cropping, landmark detection) ourselves.

Instead,  
**the processed dataset (frames + landmark sequences) was kindly provided to us by an MSc student in our supervisor’s research group**, who had previously prepared these features for internal lab experiments.

We adapted these processed inputs and integrated them into our own pipeline (dataset loaders, batching, normalization, model input formatting).

### Dataset layout (local structure used in our experiments)

```text
data/
├── frames/
├── landmarks/
└── labels/
````

To run the code, you must request access to the dataset from ChaLearn and prepare compatible frame + landmark files.

---

## 4. Model Configuration and Training Setup

### Model Parameters

* **Backbone:** ViT-based architecture adapted from S2D
* **Input resolution:** 112×112 (downscaled to avoid GPU overflow)
* **Embedding dimension:** 768
* **Hidden dimension:** 8
* **Transformer depth:** 12
* **Number of heads:** 12
* **Dropout:** 0.5

### Training Setup

* **Epochs:** 50
* **Batch size:** 32
* **Optimizer:** AdamW
* **Learning rate:** 1e-5 for both MCP/TMA and ViT backbone
* **Split:** 60% train / 20% validation / 20% test

These settings reflect a compromise between model capacity and available computation.

---

## 5. Code Structure

```text
./
├── run.py                      # entry point for training and evaluation
├── engine_for_finetuning.py    # training/validation/testing loops and metric computation
├── model_finetuining.py        # ViT + S2D model definition and regression head
├── sdl.py                      # S2D feature fusion modules (MCP/TMA-inspired blocks)
├── visualdataset.py            # dataset loader for video frame sequences
├── landmarks/                  # utilities for loading pre-extracted facial landmark images
├── datasets/                   # helper classes for building dataset splits
└── utils/                      # misc utilities (logging, mixup, metrics, etc.)
              
```

---

## 6. Installation

### Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### Create environment

```bash
conda create -n personality python=3.10
conda activate personality
```

### Install dependencies

```bash
pip install -r requirements.txt

```

---

## 7. How to Run

The main entry point is:

```bash
python run.py \
  --train \
  --data_dir /path/to/data \
  --output_dir outputs/experiment_01

## 7.1. Data Availability Notice

The training code in this repository is fully functional; however, **it cannot be executed without the preprocessed version of the ChaLearn First Impressions dataset** (video frames + facial landmarks).

We do **not** publish these processed files because:

- they were prepared internally by an MSc student in our research group,
- they are derived from the ChaLearn dataset, which cannot be redistributed,
- sharing them would violate dataset licensing terms.

### What you *can* access:
- The **raw ChaLearn First Impressions videos** and personality annotations (publicly available on the ChaLearn website)

### What you need to run the code:
- A compatible set of:
  - extracted video frames  
  - corresponding facial landmark sequences  
  - label files

You may generate your own frame sequences and landmark files from the raw dataset, or adapt the data pipeline to your preferred preprocessing method.

This repository therefore serves primarily as a **research and code demonstration** of the model architecture, training pipeline, and evaluation system.

```

Internally, `run.py`:

* builds the model using `model_finetuining.py` and `sdl.py`
* loads data from `visualdataset.py`, `datasets/`, and `landmarks/`
* runs training + validation + test via `engine_for_finetuning.py`

Argument names depend on your specific implementation of `run.py`.

---

## 8. Evaluation Pipeline & Metrics

Evaluation functions are implemented in:

* `validation_one_epoch(...)`
* `final_test(...)`

Both functions:

* run inference on validation/test sets
* save predictions and ground-truth
* compute regression metrics
* generate scatter and residual plots
* export CSV reports

### Metrics reported

* **MSE**
* **MAE**
* **RMSE**
* **R²**
* **MA (Mean Accuracy)** = `1 - |y_pred - y_true|`
* **CCC** (Concordance Correlation Coefficient)

Metrics are computed:

* **overall**, and
* **per trait** (O, C, E, A, N)

---

## 9. Limitations

* **Reduced resolution (112×112)** limits fine facial detail extraction
* **Model depth** and hidden size constrained by GPU availability
* **Training budget** restricted → fewer epochs, limited tuning
* **Only visual features** used (no audio or text modalities)

Thus, performance is reasonable and competitive, but still below large-scale SOTA systems.

---

## 10. Future Work

* Training at higher resolutions (224×224)
* Deeper ViT/S2D backbones
* Larger compute budget and systematic hyperparameter search
* Multimodal extension (audio, transcripts)


---

## 11. Contributors

This project was developed collaboratively by two team members as part of our BSc thesis.
Most components—including model implementation, data handling, and experiments—were co-developed.

**Atefeh Alimohammadi**

* Co-developed the model architecture and fine-tuning pipeline
* Implemented and adapted parts of the training/validation/testing engine
* Helped integrate visual frames and landmark inputs
* Collaborated on interpreting results and writing the thesis report

**Faezeh Salehi**

* Co-developed the data pipeline and model components
* Contributed to debugging and experiment setup
* Assisted with preprocessing integration and evaluation scripts
* Contributed to experiments, analysis, and documentation


---

## 12. Contact

For questions regarding the project:

**Name:** Atefeh Alimohammadi
**Email:** Atefehalimohammadi00163@gmail.com
**GitHub:** [https://github.com/A00163](https://github.com/A00163)

```
