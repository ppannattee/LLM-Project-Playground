# LittleParrot 🦜: A Basic AI Assistant Trained Using Supervised Fine-Tuning

**LittleParrot** 🦜 is a basic AI assistant trained using **Supervised Fine-Tuning (SFT)**. It is based on **the SmolLM-2-135M**, fine-tuned with a small set of instruction-following data. This project explores how fine-tuning a small language model with limited instruction data can enhance its ability to perform basic instructional tasks.

## Project Structure
The source code is organized into the following files within the `src` folder:

- **`LittleParrot.ipynb`**: This Jupyter notebook contains the implementation of LittleParrot. It includes code for preprocessing the dataset, fine-tuning the model using SFT, and running inference for the demonstration.

- **`SmolLM-2-135M.ipynb`**: This notebook provides code to run inference on the SmolLM-2-135M base model, which has not been fine-tuned with SFT. This serves as a baseline for evaluating the performance improvements achieved through SFT.
  
- **`SmolLM-2-135M-Instruct.ipynb`**: This notebook shows inference code for SmolLM-2-135M-Instruct, a version of SmolLM-2-135M that has been fully fine-tuned with a large instruction-following dataset. It serves as a reference for comparing LittleParrot's performance.

## Model Architecture and Dataset

LittleParrot is based on the SmolLM-2-135M model, a compact language model released by Hugging Face.

To create LittleParrot, the base pretrained SmolLM-2-135M model was fine-tuned using the [Databricks Dolly 15K (DD15K) dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k), which contains ~15,000 instruction-response pairs. This dataset is smaller and more manageable than the one used for **SmolLM-2-135M-Instruct**, which was fine-tuned with the Smol-Smoltalk dataset—a much larger set of over 460,000 instruction-response pairs.

Because of hardware limitations, fine-tuning with a larger dataset like [Smol-Smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) was not feasible in this project, making DD15K a practical choice for this exploration of SFT.

---

## Experimental Results

The fine-tuning process was conducted for **3 epochs**, with a **physical batch size of 4** and **16 gradient accumulation steps**, resulting in a **total batch size of 64**.

### Comparison of LittleParrot vs. SmolLM-2-135M (Base Model, No SFT)

Below are some example outputs from LittleParrot compared to the SmolLM-2-135M model (without SFT):

**Propt**: Who was the first woman to win a Nobel Prize?

**LittleParrot 🦜**:
```
Marie Skłodowska-Curie (1867 – 1934) was a Polish-American physicist and chemist who won the Nobel Prize in Physics in 1903 for her work on radioactivity (✅). She was the only person to win two Nobel Prizes. ...
```

**SmolLM-2-135M (general prompt format)**:

```
Nobel Prize in Physiology or Medicine 1903

What was the name of the man who won the Nobel Prize in physiology or medicine for his discovery of penicillin? ...
```

**SmolLM-2-135M (chat teplate)**:

```
What was the name of the woman who won the Nobel Prize in 1902 for her work on the microscope?Mary Anning
What did Mary Anning
```

###  LittleParrot vs. SmolLM-2-135M-Instruct

Below are example outputs from LittleParrot compared to the [SmolLM-2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct##training)

***


**Propt**: Who was the first woman to win a Nobel Prize?

**LittleParrot 🦜**:
```
Marie Skłodowska-Curie (1867 – 1934) was a Polish-American physicist and chemist who won the Nobel Prize in Physics in 1903 for her work on radioactivity (✅). She was the only person to win two Nobel Prizes. ...
```

**SmolLM-2-135M-Instruct**:
```
The Nobel Prize in Physiology or Medicine was first awarded in 1903 to Marie Skłodowska Curie for her work on radioactivity (✅). However, it wasn't until the 20th century that women like Marie Curie made significant contributions to the field of medicine.
```

***

**Propt**: What planet is closest to the sun?

**LittleParrot 🦜**:
```
Mercury is the closest planet to the Sun (✅). It takes Mercury 88 days to orbit the Sun, making it the shortest day of the year. It is also the smallest of the eight planets. The other planets are Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. Mercury is the only planet in the Solar System that does not have a moon. ...
```

**SmolLM-2-135M-Instruct**:

```
The closest star to the Sun is Proxima Centauri (❌), which is approximately 4.27 light-years away from Earth. Proxima is a red dwarf star, meaning it's smaller and cooler than the Sun, and it orbits around the center of the Milky Way galaxy at a distance of about 26,000 astronomical units (AU) from the Sun.
```

***

**Propt**: What is the capital of Canada?

**LittleParrot 🦜**
```
Toronto is Canada's capital and largest city (❌). It is located on the east bank of the St. Lawrence River in the province of Ontario, Canada. The city is one of the most populous in North America, with a population of over 2.5 million people. Toronto is home to some of the world's most prestigious universities, including the University of Toronto, the Canadian Institute for Advanced Research (CIFAR) and the Institute of Electrical and Electronics Engineers (IEEE). Toronto is also home to many of the country's most famous landmarks, such as the Eiffel Tower, the White House, and the Canadian Pacific Railway. Toronto has a rich cultural history dating back to the early 19th century, when it was known as the "Paris of the North". Today, Toronto is a global city with a diverse range of cultures and traditions, as well as a vibrant music scene and a thriving arts and entertainment scene. ...
```

**SmolLM-2-135M-Instruct**:
```
Canada is a country located in the northern part of North America. It is bordered by the Atlantic Ocean to the east, the Pacific and Arctic Oceans to the west, and the United States to the south. Canada's capital city is Ottawa (✅), and its largest city is Toronto.
```

---

## Discussion 

The comparison between **LittleParrot** and the base model of **SmolLM-2-135M** (which has not been fine-tuned with instruction data) highlights the significant improvement that SFT brings to a pretrained language model’s ability to handle instruction-based prompts.

After SFT, LittleParrot's responses demonstrated a noticeable improvement in quality. The base model, without instruction fine-tuning, struggled to generate effective and contextually appropriate answers. This demonstrates how SFT helps small models follow instructions more reliably, enhancing their performance for basic instructional tasks.

Comparing **LittleParrot** with **SmolLM-2-135M-Instruct** showed that, while the instruction-tuned model produced more concise and direct responses, LittleParrot tended to be more verbose, occasionally offering redundant information. Additionally, LittleParrot sometimes struggled to correctly terminate its sentences or include end-of-sequence (EOS) tokens.

These limitations may stem from the relatively small instruction-following dataset used for fine-tuning. SmolLM-2-135M-Instruct benefits from a much larger dataset and the application of Direct Preference Optimization (DPO) after fine-tuning, which further aligns the model's responses with human preferences and improves conciseness.

---
