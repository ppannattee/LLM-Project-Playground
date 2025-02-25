# Project 01: Mini-GPT from Scratch  

This project implements **Mini-GPT**, a GPT-style model built from scratch using **PyTorch**. It is trained on the **WikiText-103** dataset for **causal language modeling (CLM)**, a common task for text generation models.  

## Project Structure  

The source code is organized in the `src` folder, with the following key files:  

- **`model.py`** – Implements the Mini-GPT architecture.  
- **`train.py`** – Handles model training.  
- **`clm_dataset.py`** – Prepares the dataset and defines a PyTorch `Dataset` class for training.  
- **`evaluation.py`** – Contains evaluation utilities, including:  
  - **`plot_loss()`** – Plots the training loss over time.  
  - **`demo_inference()`** – Generates text using the trained model.  


## Model Architecture

Mini-GPT closely follows the architecture of **GPT-2 (small version)**, a **decoder-only** transformer model. However, due to hardware constraints, Mini-GPT is designed with a **shorter sequence length (128 tokens)** compared to the **1024 tokens** in the original GPT-2.

### Mini-GPT Model Parameters

| Parameter               | Mini-GPT | GPT-2 (Small) |
|-------------------------|---------|--------------|
| **Number of Layers**    | 12      | 12           |
| **Embedding Dimension** | 768     | 768          |
| **Number of Attention Heads** | 12 | 12         |
| **Sequence Length**     | 128     | 1024         |
| **Feedforward Hidden Size** | 3072 | 3072       |

---

## Dataset and Preprocessing  

Mini-GPT is trained on the **WikiText-103** dataset, a collection of high-quality Wikipedia articles with **~100 million tokens**. While it’s much smaller than the datasets used for training large-scale language models, it provides a reasonable starting point for building and experimenting with a transformer-based causal language model.  

Originally, WikiText-103 is divided into **train, validation, and test** sets. However, to maximize the available training data, all partitions were **merged** into a single dataset before tokenization.  

The preprocessing steps include:  

1. **Tokenization**: The text is tokenized using the **pretrained GPT-2 tokenizer** based on **Byte-Pair Encoding (BPE)** to ensure compatibility with standard GPT-style models.  
2. **Chunking**: The tokenized text is split into **fixed-length sequences of 128 tokens**, ensuring a consistent input format for training.  

This setup allows efficient training while keeping enough context in each sequence for the model to learn meaningful patterns.

---

## Experimental Results

Mini-GPT was trained for **3 epochs** with a **batch size of 16**. The training strategy closely follows that of GPT-2, including the use of the **AdamW optimizer** with a **cosine learning rate schedule** and **warmup learning rate**.  

![Training Loss Over Iterations](https://github.com/ppannattee/LLM-Project-Playground/blob/main/Core_LLM_Development/01-MiniGPT%20from%20Scratch/src/training_loss_plot.png?raw=true)

During training, the loss curve showed a steady decline, indicating that the model was learning effectively. Below are some example outputs generated using **greedy search**:  

### Sample Outputs  

#### **Prompt:** "The Great Wall of China was"  
**Generated Text:**  
The Great Wall of China was constructed to the north of the city. The city was built in the late 19th century and was the site of the first Chinese settlement in the world. The city was the capital of the Chinese Empire and was the capital of the Chinese Empire. The city was the capital of the Chinese Empire and was the capital of the Chinese Empire...  

#### **Prompt:** "In 1492, Christopher Columbus sailed"  
**Generated Text:**  
In 1492, Christopher Columbus sailed for the East Indies, where he was sent to the East Indies.  
= = = = East Indies = = = =  
In the meantime, the East Indies had been in the hands of the East Indies, and the East Indies had been in the hands of the East Indies. The East Indies had been in the hands of the East Indies, and the East Indies had been in the hands of the East Indies, and had been in the hands of the East Indies...  

#### **Prompt:** "In medieval Europe, knights were known for"  
**Generated Text:**  
In medieval Europe, knights were known for their own good, and the king's own son, the Duke of York, was the first king to be crowned. The king's father, William, was the first king to be crowned king. The king's father, William, was the first king to be crowned king. The king's father, William, was the first king to be crowned king...  

---

## Discussion 

While the generated text follows some basic **grammatical and structural patterns**, it quickly becomes repetitive. This suggests that the model **struggles with long-term coherence**, which is expected given the limited dataset and small model size.  

Compared to GPT-2, Mini-GPT's performance is significantly weaker. A major factor is the dataset size: WikiText-103 contains **only ~100 million tokens**, whereas GPT-2 was trained on OpenWebText, which has **over 9 billion tokens**. The lack of diverse and large-scale training data limits Mini-GPT’s ability to generate fluent and meaningful long-form text.  

---