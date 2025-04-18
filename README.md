# LLM Project Playground

This repository will be a collection of hands-on projects designed to deepen my understanding of Large Language Models (LLMs). I plan to use each project to explore different aspects of LLMs, ranging from fundamental concepts to advanced techniques.

Since I’m working with a single NVIDIA RTX 3080 (10GB VRAM), all projects will be optimized for local execution, utilizing compact models and efficient methods to balance performance and feasibility.

As I continue learning, I will expand this repository with new ideas, implementations, and insights, documenting my progress along the way.

---

## Project List

Below is a list of projects, each accompanied by its current status, indicated by emojis: ✅ (completed), 💡 (in progress), or 📑 (planned).

- **Mini-GPT from Scratch** ✅  
[[Project Details](https://github.com/ppannattee/LLM-Project-Playground/blob/main/projects/Mini-GPT/description.md)]

  - Implements **Mini-GPT**, a simplified GPT-style architecture built from scratch using **PyTorch**, trained on the **WikiText-103** dataset for **Causal Language Modeling (CLM)**.

- **TinyTeller: A GPT-2 Children's Short Story Generator** ✅  
  [[Project Details](https://github.com/ppannattee/LLM-Project-Playground/blob/main/projects/TinyTeller/description.md)]
  
  - Implements **TinyTeller**, a fine-tuned **GPT-2 (small)** model for generating short stories for children, using the **TinyStories dataset**.
  
- **LittleParrot 🦜: A Basic AI Assistant Trained Using Supervised Fine-Tuning** ✅  
  [[Project Details](https://github.com/ppannattee/LLM-Project-Playground/blob/main/projects/LittleParrot/description.md)]

  - Implements **LittleParrot** 🦜, an AI assistant fine-tuned with **Supervised Fine-Tuning (SFT)** on **the SmolLM-2-135**.

- **LittleParrot+ 🦜🦜: Enhanced with LoRA for Greater Performance** 💡  
  - Presents **LittleParrot+** 🦜🦜, an improved version of the original assistant, fine-tuned with **SFT** using **LoRA**.  
  - Utilizes a larger base model, **SmolLM-2-360**, while maintaining comparable training time.

- **Simple RAG from scratch** 📑