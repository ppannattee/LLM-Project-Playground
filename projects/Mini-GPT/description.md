# Mini-GPT from Scratch  

This project implements **Mini-GPT**, a simplified version of the GPT-style architecture, built from scratch using **PyTorch**. The model is trained on the **WikiText-103** dataset for **causal language modeling (CLM)**, a common task for text generation.

### Project Structure

The source code is organized into the following key files within the `src` folder:

- **`model.py`**: Implements the architecture of Mini-GPT, defining the transformer blocks, attention layers, and the overall network structure.
- **`train.py`**: Handles the training pipeline, including model initialization, loss function, optimizer setup, and training loop.
- **`clm_dataset.py`**: Prepares the **WikiText-103** dataset for training, defining a custom PyTorch `Dataset` class to load and preprocess the data.
- **`evaluation.py`**: Provides utilities for evaluating the trained model, including:
  - **`plot_loss()`**: A function to visualize the training loss over epochs.
  - **`demo_inference()`**: A simple script to generate text using the trained Mini-GPT model, demonstrating its text generation capabilities.

**Note**: Many parts of the code are inspired by Andrej Karpathy’s repository for [implementing GPT-2 from scratch](https://github.com/karpathy/build-nanogpt), which serves as an excellent reference for understanding the core concepts behind transformer-based language models.

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

Mini-GPT was trained for **3 epochs** using a **batch size of 16**, with **gradient accumulation** applied in a factor of 4 to achieve a **total effective batch size of 64**. The training process follows a strategy similar to that of GPT-2, such as **AdamW optimizer**, **Cosine learning rate schedule**, and **Learning rate warmup**.

![Training Loss Over Iterations](https://github.com/ppannattee/LLM-Project-Playground/blob/main/projects/Mini-GPT/src/training_loss_plot.png?raw=true)

During training, the loss curve showed a steady decline, indicating that the model was learning effectively. Below are some example outputs generated using **greedy search**:  

### Sample Outputs  

**Prompt:**
```
In the 18th century, scientists discovered that
```

**Generated Text:**
```  
In the 18th century, scientists discovered that the planet was a giant planet , and that the planet was a giant planet . The planet was discovered in 1869 by the French astronomer Jean @-@ Jacques Leclerc , who had discovered the planet in 1869 . Leclerc had discovered the planet in 1869...
```

---

**Prompt:**
```
The Industrial Revolution led to significant advancements in the production of
```

**Generated Text:**
```
The Industrial Revolution led to significant advancements in the production of the textile industry .
 = = = Industrial Revolution = = =
 The Industrial Revolution brought a new wave of economic growth and the growth of the textile industry . The Industrial Revolution brought about a new generation of textile production to the United States . The Industrial Revolution brought about a new generation of textile production to the United States . The Industrial Revolution brought about a new generation of textile production to the United States...
```
---

**Prompt:**
```
The theory of evolution by natural selection was proposed by
```

**Generated Text:**  
```
The theory of evolution by natural selection was proposed by the American biologist William Gould in 1851 . Gould 's theory was based on the idea that evolution was a natural selection . Gould 's theory was based on the idea that evolution was a natural selection...
```
---

## Discussion 

While the generated text follows some basic **grammatical and structural patterns**, it quickly becomes repetitive. This suggests that the model **struggles with long-term coherence**, which is expected given the limited dataset and small model size.  

Compared to GPT-2, Mini-GPT's performance is significantly weaker. A major factor is the dataset size: WikiText-103 contains **only ~100 million tokens**, whereas GPT-2 was trained on OpenWebText, which has **over 9 billion tokens**. The lack of diverse and large-scale training data limits Mini-GPT’s ability to generate fluent and meaningful long-form text.  

---