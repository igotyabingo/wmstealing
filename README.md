# LLM Watermark Stealing — JSV & DeMark Baselines

This repository reproduces representative watermark-stealing methodologies—[**JSV**](https://github.com/eth-sri/watermark-stealing) and [**DeMark**](https://arxiv.org/pdf/2410.13808)—and contains the code and scripts used to run scrubbing (removal) and spoofing (exploitation) experiments.

---

## 📦 Environment Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/igotyabingo/wmstealing 
cd wmstealing
```
### 2️⃣ Set Environment Variables
```bash
# Copy the example file and edit it to create your local `.env`:

cp .env_example .env
```

### 3️⃣ Create Conda Environment
```bash
# create environment with miniconda

conda create -n ws python=3.11.13
conda activate ws
pip install -r requirements.txt
```

## 🚀 Experiment Pipeline
### ▶️ Generate Target Model Outputs
```bash
# generate target LLM's output (base + watermarked) for JSV

bash ./scripts/jsv/run_generate_output.sh
```
### ⚙️ Run Attacks for JSV
```bash
# spoofing attack
bash ./scripts/jsv/run_spoofing.sh

# scrubbing attack
bash ./scripts/jsv/run_scrubbing.sh

# evaluation
bash ./scripts/jsv/evaluate.sh
```
### ⚙️ Run Attacks for DeMark
```bash
# spoofing attack
bash ./scripts/demark/run_exp_watermark_exploitation.sh

# scrubbing attack
bash ./scripts/demark/run_exp_watermark_removal_KGW.sh

# evaluation
bash ./scripts/demark/evaluate.sh
```


## ⚗️ Experiment Settings
### Watermark Parameters
We adopt the same watermarking parameters as described in each original paper.
- **JSV**: γ = 0.25, δ = 4.0
- **DeMark**: γ = 0.5, δ = 2.0

### Evaluation Datasets
We evaluated on three datasets: `mmw_book_report` and `dolly_cw` (primary evaluation datasets), and `c4_test`, which is used only for spoofing attacks.

### Attack Configurations
- For **scrubbing** experiments,
**JSV** uses `Llama-3.1-8B`, `Llama-3.2-3B`, and `Mistral-7B` as target models, and `dipper-paraphraser-xxl` as the surrogate model.
**DeMark**, in contrast, employs `Llama-3.2-3B` as both the target and surrogate model.
- For **spoofing** experiments, the following (target model → surrogate model) pairs are used:
  - `Llama-3.1-8B` → `Llama-3.2-3B`
  - `Llama-3.2-3B` → `Llama-3.2-1B`
  - `Qwen3-4B` → `Qwen2.5-7B`

### Evaluation Metrics
<div align="center"> <table style="margin-left:auto; margin-right:auto; text-align:center; border-collapse:collapse;"> <thead> <tr> <th style="padding:6px 12px; border:1px solid #ddd;">Metric</th> <th style="padding:6px 12px; border:1px solid #ddd;">Description</th> </tr> </thead> <tbody> <tr> <td style="padding:6px 12px; border:1px solid #ddd;">TPR@FPR</td> <td style="padding:6px 12px; border:1px solid #ddd;">10%, (5%), 1%, 0.1%</td> </tr> <tr> <td style="padding:6px 12px; border:1px solid #ddd;">Perplexity</td> <td style="padding:6px 12px; border:1px solid #ddd;">Computed using Llama-2-7B</td> </tr> <tr> <td style="padding:6px 12px; border:1px solid #ddd;">GPT-based Quality Evaluation</td> <td style="padding:6px 12px; border:1px solid #ddd;">Separate prompt templates for scrubbing & spoofing</td> </tr> </tbody> </table> </div>

- #### GPT Evaluation Prompt used for Scrubbing attack (from DeMark paper)
```
You are given a prompt and a response, and you provide a grade out of 100 measuring the quality of the response, in terms of accuracy, level of details, and typographical, grammatical and lexical correctness. Remove points as soon as one of the criteria is missed.
Please rate the response on a scale of 1 to 100 by strictly following this format: "Rating: [[51]]".
```
- #### GPT Evaluation Prompt used for Spoofing attack (from JSV paper)
```
{
  "Style": {
    "Explanation": "Is the text written fluently and are the grammar and sentence structure correct? Is the answer creative in its writing or direct and does this fit the question?",
    "Score": "score from 1 to 10"
  },
  "Consistency": {
    "Explanation": "Is the answer consistent both in writing and in its answer to the users question?",
    "Score": "score from 1 to 10"
  },
  "Accuracy": {
    "Explanation": "Does the answer invent any facts that are not actually true or did it give factually correct answers?",
    "Score": "score from 1 to 10"
  }
}
```
## 🧮 Results Summary
### Spoofing Attack
<table>
<tr>
  <th colspan="8" style="text-align:center;">MMW Book Report + Llama-3.1-8B</th>
</tr>
<tr>
  <th></th>
  <th>TPR@FPR=10% ↑</th>
  <th>TPR@FPR=1% ↑</th>
  <th>TPR@FPR=0.1% ↑</th>
  <th>median p-value ↓</th>
  <th>perplexity ↓</th>
  <th>GPT score ↑</th>
</tr>
<tr>
  <td>+ JSV</td>
  <td>36%</td>
  <td>10%</td>
  <td>4%</td>
  <td>0.223</td>
  <td><b>2.814</b></td>
  <td><b>6.34</b></td>
</tr>
<tr>
  <td>+ De-Mark (gray)</td>
  <td>99%</td>
  <td><b>96%</b></td>
  <td><b>88%</b></td>
  <td><b>5.73E-06</b></td>
  <td>3.77</td>
  <td>6.073</td>
</tr>
<tr>
  <td>+ De-Mark (black)</td>
  <td><b>100%</b></td>
  <td>94%</td>
  <td>73%</td>
  <td>6.93E-05</td>
  <td>3.55</td>
  <td>6.17</td>
</tr>
</table>

<table>
<tr>
  <th colspan="8" style="text-align:center;">MMW Book Report + Qwen3-4B</th>
</tr>
<tr>
  <th></th>
  <th>TPR@FPR=10% ↑</th>
  <th>TPR@FPR=1% ↑</th>
  <th>TPR@FPR=0.1% ↑</th>
  <th>median p-value ↓</th>
  <th>perplexity ↓</th>
  <th>GPT score ↑</th>
</tr>
<tr>
  <td>+ JSV</td>
  <td>32%</td>
  <td>11%</td>
  <td>4%</td>
  <td>0.209</td>
  <td><b>4.257</b></td>
  <td><b>6.843</b></td>
</tr>
<tr>
  <td>+ De-Mark (gray)</td>
  <td>98%</td>
  <td>88%</td>
  <td><b>74%</b></td>
  <td>1.10E-04</td>
  <td>5.174</td>
  <td>6.47</td>
</tr>
<tr>
  <td>+ De-Mark (black)</td>
  <td><b>100%</b></td>
  <td><b>95%</b></td>
  <td><b>74%</b></td>
  <td><b>4.32E-05</b></td>
  <td>5.145</td>
  <td>6.6</td>
</tr>
</table>

<table>
<tr>
  <th colspan="8" style="text-align:center;">MMW Book Report + Llama-3.2-3B</th>
</tr>
<tr>
  <th></th>
  <th>TPR@FPR=10% ↑</th>
  <th>TPR@FPR=1% ↑</th>
  <th>TPR@FPR=0.1% ↑</th>
  <th>median p-value ↓</th>
  <th>perplexity ↓</th>
  <th>GPT score ↑</th>
</tr>
<tr>
  <td>+ JSV</td>
  <td>70%</td>
  <td>34%</td>
  <td>18%</td>
  <td>0.036</td>
  <td><b>3.781</b></td>
  <td><b>4.5</b></td>
</tr>
<tr>
  <td>+ De-Mark (gray)</td>
  <td>98%</td>
  <td><b>96%</b></td>
  <td><b>90%</b></td>
  <td><b>1.93E-06</b></td>
  <td>4.65</td>
  <td>4.47</td>
</tr>
<tr>
  <td>+ De-Mark (black)</td>
  <td><b>100%</b></td>
  <td><b>96%</b></td>
  <td><b>90%</b></td>
  <td>3.34E-06</td>
  <td>4.681</td>
  <td>4.49</td>
</tr>
</table>

---

<table>
<tr>
  <th colspan="8" style="text-align:center;">Dolly CW + Llama-3.1-8B</th>
</tr>
<tr>
  <th></th>
  <th>TPR@FPR=10% ↑</th>
  <th>TPR@FPR=1% ↑</th>
  <th>TPR@FPR=0.1% ↑</th>
  <th>median p-value ↓</th>
  <th>perplexity ↓</th>
  <th>GPT score ↑</th>
</tr>
<tr>
  <td>+ JSV</td>
  <td>56%</td>
  <td>26%</td>
  <td>11%</td>
  <td>0.063</td>
  <td>6.622</td>
  <td><b>7.28</b></td>
</tr>
<tr>
  <td>+ De-Mark (gray)</td>
  <td>96%</td>
  <td><b>89%</b></td>
  <td><b>77%</b></td>
  <td><b>4.32E-05</b></td>
  <td>3.971</td>
  <td>7.177</td>
</tr>
<tr>
  <td>+ De-Mark (black)</td>
  <td><b>97%</b></td>
  <td>82%</td>
  <td>70%</td>
  <td>2.66E-05</td>
  <td><b>3.931</b></td>
  <td>3.307</td>
</tr>
</table>

<table>
<tr>
  <th colspan="8" style="text-align:center;">Dolly CW + Qwen3-4B</th>
</tr>
<tr>
  <th></th>
  <th>TPR@FPR=10% ↑</th>
  <th>TPR@FPR=1% ↑</th>
  <th>TPR@FPR=0.1% ↑</th>
  <th>median p-value ↓</th>
  <th>perplexity ↓</th>
  <th>GPT score ↑</th>
</tr>
<tr>
  <td>+ JSV</td>
  <td>37%</td>
  <td>11%</td>
  <td>3%</td>
  <td>0.173</td>
  <td>7.948</td>
  <td><b>7.54</b></td>
</tr>
<tr>
  <td>+ De-Mark (gray)</td>
  <td><b>99%</b></td>
  <td><b>88%</b></td>
  <td><b>70%</b></td>
  <td><b>6.93E-05</b></td>
  <td>5.293</td>
  <td>7.45</td>
</tr>
<tr>
  <td>+ De-Mark (black)</td>
  <td>95%</td>
  <td>87%</td>
  <td>69%</td>
  <td><b>6.93E-05</b></td>
  <td><b>5.22</b></td>
  <td>7.417</td>
</tr>
</table>

<table>
<tr>
  <th colspan="8" style="text-align:center;">Dolly CW + Llama-3.2-3B</th>
</tr>
<tr>
  <th></th>
  <th>TPR@FPR=10% ↑</th>
  <th>TPR@FPR=1% ↑</th>
  <th>TPR@FPR=0.1% ↑</th>
  <th>median p-value ↓</th>
  <th>perplexity ↓</th>
  <th>GPT score ↑</th>
</tr>
<tr>
  <td>+ JSV</td>
  <td>73%</td>
  <td>42%</td>
  <td>28%</td>
  <td>0.013</td>
  <td>6.933</td>
  <td><b>6.533</b></td>
</tr>
<tr>
  <td>+ De-Mark (gray)</td>
  <td>99%</td>
  <td><b>93%</b></td>
  <td>75%</td>
  <td><b>2.66E-05</b></td>
  <td><b>4.64</b></td>
  <td>6.123</td>
</tr>
<tr>
  <td>+ De-Mark (black)</td>
  <td><b>100%</b></td>
  <td>90%</td>
  <td><b>80%</b></td>
  <td><b>2.66E-05</b></td>
  <td>4.737</td>
  <td>6.12</td>
</tr>
</table>

---

<table>
<tr>
  <th colspan="8" style="text-align:center;">C4 + Llama-3.1-8B</th>
</tr>
<tr>
  <th></th>
  <th>TPR@FPR=10% ↑</th>
  <th>TPR@FPR=1% ↑</th>
  <th>TPR@FPR=0.1% ↑</th>
  <th>median p-value ↓</th>
  <th>perplexity ↓</th>
  <th>GPT score ↑</th>
</tr>
<tr>
  <td>+ JSV</td>
  <td>83%</td>
  <td>66%</td>
  <td>56%</td>
  <td>4.33-E04</td>
  <td>7.715</td>
  <td><b>6.73</b></td>
</tr>
<tr>
  <td>+ De-Mark (gray)</td>
  <td><b>98%</b></td>
  <td><b>93%</b></td>
  <td><b>85%</b></td>
  <td><b>1.03E-06</b></td>
  <td><b>7.536</b></td>
  <td>6.597</td>
</tr>
<tr>
  <td>+ De-Mark (black)</td>
  <td>97%</td>
  <td>91%</td>
  <td>83%</td>
  <td>1.51E-06</td>
  <td>7.652</td>
  <td>6.577</td>
</tr>
</table>

<table>
<tr>
  <th colspan="8" style="text-align:center;">C4 + Qwen3-4B</th>
</tr>
<tr>
  <th></th>
  <th>TPR@FPR=10% ↑</th>
  <th>TPR@FPR=1% ↑</th>
  <th>TPR@FPR=0.1% ↑</th>
  <th>median p-value ↓</th>
  <th>perplexity ↓</th>
  <th>GPT score ↑</th>
</tr>
<tr>
  <td>+ JSV</td>
  <td>53%</td>
  <td>27%</td>
  <td>8%</td>
  <td>0.091</td>
  <td><b>6.944</b></td>
  <td><b>7.59</b></td>
</tr>
<tr>
  <td>+ De-Mark (gray)</td>
  <td><b>95%</b></td>
  <td><b>83%</b></td>
  <td><b>63%</b></td>
  <td><b>1.72E-04</b></td>
  <td>7.007</td>
  <td>7.497</td>
</tr>
<tr>
  <td>+ De-Mark (black)</td>
  <td>94%</td>
  <td>77%</td>
  <td>61%</td>
  <td>2.66E-04</td>
  <td>7.253</td>
  <td>7.35</td>
</tr>
</table>

<table>
<tr>
  <th colspan="8" style="text-align:center;">C4 + Llama-3.2-3B</th>
</tr>
<tr>
  <th></th>
  <th>TPR@FPR=10% ↑</th>
  <th>TPR@FPR=1% ↑</th>
  <th>TPR@FPR=0.1% ↑</th>
  <th>median p-value ↓</th>
  <th>perplexity ↓</th>
  <th>GPT score ↑</th>
</tr>
<tr>
  <td>+ JSV</td>
  <td>87%</td>
  <td>71%</td>
  <td>60%</td>
  <td>2.17E-04</td>
  <td>8.595</td>
  <td><b>5.46</b></td>
</tr>
<tr>
  <td>+ De-Mark (gray)</td>
  <td><b>100%</b></td>
  <td><b>99%</b></td>
  <td><b>96%</b></td>
  <td><b>1.44E-07</b></td>
  <td><b>8.16</b></td>
  <td>5.05</td>
</tr>
<tr>
  <td>+ De-Mark (black)</td>
  <td>100%</td>
  <td>97%</td>
  <td>94%</td>
  <td>1.88E-07</td>
  <td>8.221</td>
  <td>5.217</td>
</tr>
</table>

---

### Scrubbing Attack
- De-Mark results are taken directly from the original publication
<table align="center">
<tr>
  <th colspan="6" style="text-align:center; width:100%;">MMW Book Report + Llama-3.1-8B</th>
</tr>
<tr>
  <th></th>
  <th>TPR@FPR=10% ↓</th>
  <th>TPR@FPR=1% ↓</th>
  <th>TPR@FPR=0.1% ↓</th>
  <th>median p-value ↑</th>
  <th>GPT score ↑</th>
</tr>
<tr>
  <td>+ JSV</td>
  <td>36%</td>
  <td>10%</td>
  <td>4%</td>
  <td><b>0.745</b></td>
  <td>81.4</td>
</tr>
<tr>
  <td>+ De-Mark</td>
  <td><b>2%</b></td>
  <td><b>1%</b></td>
  <td><b>1%</b></td>
  <td>1.78E-01</td>
  <td><b>94.63</b></td>
</tr>
</table>

<table align="center">
<tr>
  <th colspan="6" style="text-align:center; width:100%;">MMW Book Report + Llama-3.2-3B</th>
</tr>
<tr>
  <th></th>
  <th>TPR@FPR=10% ↓</th>
  <th>TPR@FPR=1% ↓</th>
  <th>TPR@FPR=0.1% ↓</th>
  <th>median p-value ↑</th>
  <th>GPT score ↑</th>
</tr>
<tr>
  <td>+ JSV</td>
  <td>10%</td>
  <td><b>2%</b></td>
  <td><b>1%</b></td>
  <td><b>0.771</b></td>
  <td>78.73</td>
</tr>
<tr>
  <td>+ De-Mark</td>
  <td><b>6%</b></td>
  <td><b>2%</b></td>
  <td>2%</td>
  <td>8.29E-02</td>
  <td><b>93.98</b></td>
</tr>
</table>

---

<table align="center">
<tr>
  <th colspan="6" style="text-align:center; width:100%;">Dolly CW + Llama-3.1-8B</th>
</tr>
<tr>
  <th></th>
  <th>TPR@FPR=10% ↓</th>
  <th>TPR@FPR=1% ↓</th>
  <th>TPR@FPR=0.1% ↓</th>
  <th>median p-value ↑</th>
  <th>GPT score ↑</th>
</tr>
<tr>
  <td>+ JSV</td>
  <td>18%</td>
  <td><b>4%</b></td>
  <td>2%</td>
  <td><b>0.58</b></td>
  <td>79.43</td>
</tr>
<tr>
  <td>+ De-Mark</td>
  <td><b>10%</b></td>
  <td>6%</td>
  <td><b>1%</b></td>
  <td>1.78E-01</td>
  <td><b>92.88</b></td>
</tr>
</table>

<table align="center">
<tr>
  <th colspan="6" style="text-align:center;">Dolly CW + Llama-3.2-3B</th>
</tr>
<tr>
  <th></th>
  <th>TPR@FPR=10% ↓</th>
  <th>TPR@FPR=1% ↓</th>
  <th>TPR@FPR=0.1% ↓</th>
  <th>median p-value ↑</th>
  <th>GPT score ↑</th>
</tr>
<tr>
  <td>+ JSV</td>
  <td>16%</td>
  <td><b>5%</b></td>
  <td><b>0%</b></td>
  <td><b>0.609</b></td>
  <td>80.13</td>
</tr>
<tr>
  <td>+ De-Mark</td>
  <td><b>9%</b></td>
  <td><b>5%</b></td>
  <td>3%</td>
  <td>9.25E-02</td>
  <td><b>91.57</b></td>
</tr>
</table>



## 📚 Citation
The code and experiments in this repository are implemented following the methods described in the following works:
```
@inproceedings{jovanovic2024watermarkstealing,
    author = {Jovanović, Nikola and Staab, Robin and Vechev, Martin},
    title = {Watermark Stealing in Large Language Models},
    journal = {{ICML}},
    year = {2024}
}

@article{chen2024mark,
  title={De-mark: Watermark Removal in Large Language Models},
  author={Chen, Ruibo and Wu, Yihan and Guo, Junfeng and Huang, Heng},
  journal={arXiv preprint arXiv:2410.13808},
  year={2024}
}
```
---
Last Updated: Nov 1, 2025