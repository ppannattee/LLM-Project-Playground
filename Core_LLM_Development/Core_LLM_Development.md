
## Category 1: Core LLM Development
Focus: Understanding and implementing transformer-based models, tokenization, and pre-training strategies.

1. **Mini-GPT from Scratch**  
   - *Description*: Build a small GPT-style model (e.g., 6 layers, 256 hidden size) with multi-head self-attention and train it on Tiny Shakespeare.  
   - *Skills*: Transformer architecture, PyTorch implementation, text generation.  
   - *Feasibility*: Small model fits in 10GB VRAM with batch size ~4-8.

2. **BERT-like Masked Language Model**  
   - *Description*: Implement a BERT-style model (12 layers, 512 hidden size) and pre-train it on a Wikipedia subset using masked language modeling (MLM).  
   - *Skills*: Bidirectional attention, tokenization, pre-training.  
   - *Feasibility*: Use gradient accumulation for larger batches.

3. **Custom Tokenizer Development**  
   - *Description*: Create a Byte Pair Encoding (BPE) or WordPiece tokenizer from scratch and train it on a small corpus (e.g., OpenWebText subset).  
   - *Skills*: Tokenization algorithms, vocabulary optimization.  
   - *Feasibility*: CPU-based preprocessing, minimal VRAM usage.

4. **Decoder-Only Transformer for Story Generation**  
   - *Description*: Build a decoder-only transformer and train it to generate coherent short stories using BookCorpus.  
   - *Skills*: Causal language modeling, dataset handling.  
   - *Feasibility*: Reduce model size (e.g., 8 layers, 384 hidden size).

5. **Tiny T5 Implementation**  
   - *Description*: Implement a small T5 model (encoder-decoder) for text-to-text tasks like summarization on CNN/Daily Mail.  
   - *Skills*: Encoder-decoder architecture, seq2seq learning.  
   - *Feasibility*: Limit to 6 layers per component, batch size ~2-4.

6. **Multi-Task Transformer**  
   - *Description*: Design a transformer that jointly trains on sentiment analysis and NER using SST-2 and CoNLL-2003.  
   - *Skills*: Multi-task learning, task-specific heads.  
   - *Feasibility*: Small model with shared backbone.

7. **Positional Encoding Variants**  
   - *Description*: Experiment with rotary positional embeddings (RoPE) vs. sinusoidal embeddings in a small transformer.  
   - *Skills*: Positional encoding, attention mechanisms.  
   - *Feasibility*: Minimal VRAM overhead.

8. **Attention Mechanism Playground**  
   - *Description*: Implement and compare standard self-attention, multi-head attention, and sparse attention (e.g., Longformer-style).  
   - *Skills*: Attention optimization, sparse computation.  
   - *Feasibility*: Small sequences, batch size ~8.

9. **Pre-trained Model Fine-Tuning**  
   - *Description*: Fine-tune a distilled BERT (e.g., DistilBERT) on IMDb reviews for sentiment classification.  
   - *Skills*: Transfer learning, fine-tuning strategies.  
   - *Feasibility*: Use Hugging Face’s distilled models.

10. **Dynamic Vocabulary Transformer**  
    - *Description*: Build a transformer that adapts its vocabulary size during training based on token frequency.  
    - *Skills*: Dynamic embeddings, vocabulary pruning.  
    - *Feasibility*: Small model and dataset.

11. **Layer-wise Learning Rate Transformer**  
    - *Description*: Implement a transformer with layer-wise learning rates and test on text classification.  
    - *Skills*: Optimization techniques, transformer training.  
    - *Feasibility*: Fits with gradient accumulation.

12. **Conditional Text Generation**  
    - *Description*: Train a transformer to generate text conditioned on sentiment (positive/negative) using SST-2.  
    - *Skills*: Conditional generation, control mechanisms.  
    - *Feasibility*: Small model, batch size ~4.

13. **Lightweight Encoder-Decoder for Translation**  
    - *Description*: Build a small transformer for English-to-French translation using a WMT subset.  
    - *Skills*: Seq2seq modeling, bilingual training.  
    - *Feasibility*: Limit to 6 layers, small batch size.

14. **Self-Attention Visualization Tool**  
    - *Description*: Create a tool to visualize self-attention weights in a trained transformer on sample sentences.  
    - *Skills*: Visualization, attention mechanics.  
    - *Feasibility*: Post-training analysis, low VRAM.

15. **Knowledge Distillation of BERT**  
    - *Description*: Distill a pre-trained BERT into a smaller model (e.g., 4 layers) using a teacher-student framework.  
    - *Skills*: Knowledge distillation, model compression.  
    - *Feasibility*: Use pre-trained BERT, small student model.

16. **Causal Language Model with Prefix Tuning**  
    - *Description*: Implement prefix tuning on a small GPT-style model for task-specific generation.  
    - *Skills*: Parameter-efficient tuning, text generation.  
    - *Feasibility*: Minimal VRAM for prefixes.

17. **Token Dropout Transformer**  
    - *Description*: Add token-level dropout to a transformer’s input embeddings and evaluate generalization.  
    - *Skills*: Regularization, transformer robustness.  
    - *Feasibility*: Small model tweak.

18. **Subword Embedding Exploration**  
    - *Description*: Compare subword embeddings (BPE vs. WordPiece vs. SentencePiece) in a small transformer on NER.  
    - *Skills*: Embedding strategies, tokenization impact.  
    - *Feasibility*: Preprocessing focus, small model.

19. **Tiny LLaMA Clone**  
    - *Description*: Replicate a scaled-down LLaMA-style model (e.g., 6 layers, 512 hidden size) and train on OpenWebText.  
    - *Skills*: Modern LLM architecture, pre-training.  
    - *Feasibility*: Reduce parameters to fit VRAM.

20. **Hybrid RNN-Transformer Model**  
    - *Description*: Combine an LSTM with a transformer decoder for text generation and compare to pure transformer.  
    - *Skills*: Hybrid architectures, RNN integration.  
    - *Feasibility*: Small LSTM + transformer.

---