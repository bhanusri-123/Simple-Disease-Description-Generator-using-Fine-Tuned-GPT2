

# Disease Description Generation using nanoGPT

## Overview

This project demonstrates how to fine-tune a pretrained GPT-2 model using **nanoGPT** on a custom medical dataset containing **disease names and their corresponding descriptions**.

The goal is to enable the language model to learn relationships between disease names and their textual descriptions so that it can generate meaningful disease-related content after fine-tuning.

---

## Project Objectives

* Understand the complete GPT training pipeline.
* Learn how transfer learning works using GPT-2.
* Fine-tune a pretrained language model on a custom dataset.
* Compare text generation before and after fine-tuning.
* Visualize training progress using loss graphs.

---

## Dataset

The dataset consists of:

* **Disease Name**
* **Disease Description**

Example:

| Disease      | Description                                                                      |
| ------------ | -------------------------------------------------------------------------------- |
| Diabetes     | A chronic disease in which the body cannot properly regulate blood sugar levels. |
| Asthma       | A respiratory condition that causes inflammation and narrowing of the airways.   |
| Hypertension | A condition characterized by persistently high blood pressure.                   |

During preprocessing, the disease name and description are combined into plain text and tokenized before training.

---

## Project Structure

```
nanogpt/
│
├── configs/                  # Training configuration files
│
├── data/
│   ├── disease_dataset/      # Dataset files
│   ├── prepare.py            # Dataset preparation
│   └── ...
│
├── scripts/
│   ├── prepare_data.py       # Dataset preprocessing
│   ├── train.py              # Train model from scratch
│   ├── finetune.py           # Fine-tune GPT-2
│   ├── sample.py             # Generate text
│   ├── compare.py            # Compare pretrained vs fine-tuned output
│   └── plot_loss.py          # Plot loss graph
│
├── src/
│   └── nanogpt/
│       ├── model.py
│       ├── trainer.py
│       ├── builder.py
│       ├── generate.py
│       ├── data.py
│       ├── config.py
│       └── common.py
│
└── out/                      # Saved checkpoints
```

---

## Workflow

### 1. Prepare the Dataset

The dataset is read and converted into a format suitable for GPT training.

Outputs generated:

* `train.bin`
* `val.bin`
* `meta.pkl`

Command:

```bash
uv run python scripts/prepare_data.py disease_dataset
```

---

### 2. Fine-tune GPT-2

Load pretrained GPT-2 weights and continue training using the disease dataset.

Command:

```bash
uv run python scripts/finetune.py --config configs/finetune_gpt2.toml
```

---

### 3. Resume Training (Optional)

If training stops, continue from the saved checkpoint.

```bash
uv run python scripts/finetune.py --config configs/finetune_gpt2.toml
```

---

### 4. Generate Text

Generate disease-related text using the fine-tuned model.

```bash
uv run python scripts/sample.py \
--out_dir out-finetune \
--start "Diabetes:"
```

---

### 5. Compare Results

Compare the output from:

* Original pretrained GPT-2
* Fine-tuned GPT-2

This demonstrates how fine-tuning changes the model's writing style and domain knowledge.

---

### 6. Plot Training Loss

Visualize training and validation loss.

```bash
uv run python scripts/plot_loss.py
```

---

## Model Architecture

The project uses the GPT architecture consisting of:

* Token Embedding Layer
* Positional Embedding Layer
* Transformer Blocks
* Multi-Head Self Attention
* Feed Forward Network (MLP)
* Layer Normalization
* Language Modeling Head

---

## Training Process

1. Load dataset
2. Tokenize text
3. Create train/validation split
4. Load GPT-2 pretrained weights
5. Perform forward propagation
6. Compute Cross Entropy Loss
7. Perform backpropagation
8. Update weights using AdamW optimizer
9. Save checkpoints
10. Generate text

---

## Outputs

The project produces:

* Fine-tuned GPT-2 model
* Training checkpoints
* Generated disease descriptions
* Training loss graph
* Validation loss graph

---

## Technologies Used

* Python
* PyTorch
* nanoGPT
* GPT-2
* NumPy
* tiktoken
* Matplotlib

---

## Learning Outcomes

* GPT architecture
* Transformer fundamentals
* Fine-tuning pretrained language models
* Dataset preprocessing
* Tokenization
* Checkpointing
* Text generation
* Loss visualization
* End-to-end language model training workflow

