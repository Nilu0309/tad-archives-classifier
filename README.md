# TaD Archives Classifier — NLP Coursework Project

### Overview
This repository contains my **Text-as-Data (TaD)** NLP coursework project for **museum record classification**.  
The goal is to automatically classify short archival text descriptions into one of **five UK heritage institutions** using data cleaning, exploratory analysis, prompt-based classification with LLMs, and fine-tuned transformer models.

---

### Problem Definition
Each record in the dataset corresponds to a museum or archival item that must be assigned to one of the following institutions:

| Label | Institution |
|:------|:-------------|
| 0 | National Maritime Museum |
| 1 | National Railway Museum |
| 2 | Royal Botanic Gardens, Kew |
| 3 | Royal College of Physicians of London |
| 4 | Shakespeare Birthplace Trust |

The dataset was provided via the TaD coursework link (`tinyurl.com/tadarchives`) and split into **train**, **validation**, and **test** subsets.

---

## 1. Data Cleaning (Q1)
The raw dataset contained:
- Non-uniform field names (`"labl"`, `"key"`, `"description"`, etc.)
- Missing or malformed labels
- Non-string content fields
- Spelling variations in institution names

**Fixes applied:**
- Standardized keys → `{"id", "label", "content"}`
- Used *fuzzy string matching* to remap incorrect labels to the five valid institutions
- Dropped missing and non-string content rows (2 missing, 3 numeric-only)
- Validated final structure: 150 training, 50 validation, 50 test samples

---

## 2. Data Exploration (Q2)
- Dataset splits: **Train = 60%**, **Val = 20%**, **Test = 20%**  
- Character lengths:  
  - Train: *163 – 4263*  
  - Val: *154 – 2794*  
  - Test: *167 – 3479*

Top tokens per class (via spaCy lemmatization & stopword removal):

| Class | Frequent Tokens |
|:------|:----------------|
| 0 National Maritime Museum | john, sir, henry, enclosure, james |
| 1 National Railway Museum | 2000, 7200, 756, gb, collection |
| 2 Royal Botanic Gardens, Kew | letter, include, paper, list, contain |
| 3 Royal College of Physicians | seal, common, mr., college, fellow |
| 4 Shakespeare Birthplace Trust | mr., work, bill, account, letter |

These insights informed tokenization and truncation limits for transformer inputs.

---

## 3. Prompting with an LLM (Q3)
Three prompt templates were evaluated using **Llama-3.1-8B-Instruct**, producing 150 predictions each.  
Invalid outputs were grouped into a sixth “class 5”.

| Template | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|:----------|:---------|:----------------|:--------------|:----------|
| A | 0.00 | 0.00 | 0.00 | 0.00 |
| B | **0.76** | **0.79** | **0.75** | **0.74** |
| C | 0.71 | 0.67 | 0.58 | 0.58 |

**Result:** Template B was the most effective, producing valid structured outputs and balanced scores.

---

## 4. Fine-Tuning a Transformer (Q4)
Model: **bert-base-uncased**  
Hyperparameters: *8 epochs*, *lr = 5e-5*, *batch = 8*  

Despite steady training-loss decline, validation loss rose sharply → **severe overfitting** on the small dataset (150 samples).  
Final validation accuracy: **24%**, dominated by a single class prediction.

---

## 5. Validation Set Issue (Q5)
A confusion matrix revealed mirrored class alignment — validation labels were **reversed**.  
Fix:  
```python
mapping = {0:4, 1:3, 2:2, 3:1, 4:0}
```
After relabeling, performance and diagonal structure were restored.

---

## 6. Hyperparameter & Model Tuning (Q6)

Four transformer models were fine-tuned under identical training configurations to compare their performance and generalization ability:

| Model | Accuracy | Macro F1 | Notes |
|:-------|:----------:|:----------:|:------|
| **BERT (bert-base-uncased)** | 94% | 94.24% | Strong and stable performance, good balance across classes |
| **RoBERTa (roberta-base)** | 92% | 91.00% | Slightly unstable early epochs and minor overfitting in later stages |
| **DistilBERT (distilbert-base-uncased)** | **96%** | **96.26%** | Best overall performance; fast convergence and efficient |
| **BiomedBERT (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract)** | 86% | 84.25% | Underperformed due to domain mismatch (biomedical text pretraining) |

### Key Insights
- **DistilBERT** achieved the highest overall accuracy and macro F1-score while being computationally lighter than other models.  
- **BERT** remained a strong alternative with balanced metrics.  
- **RoBERTa** showed potential but suffered from early instability and possible overfitting.  
- **BiomedBERT** was not suitable for this dataset, as it’s specialized for biomedical abstracts.

**Conclusion:**  
For this classification task, **DistilBERT** is the optimal model, combining efficiency, speed, and accuracy, making it ideal for real-world deployment.

---

## 7. Final Evaluation & Deployment (Q7)

After hyperparameter tuning, **DistilBERT** was selected as the final model based on its validation macro-F1 score (96.26%).  
The model was then evaluated on the **test set** to assess generalization performance.

### Test Results
| Metric | Score |
|:--------|:------:|
| **Accuracy** | 90.00% |
| **Macro Precision** | 89.75% |
| **Macro Recall** | 92.31% |
| **Macro F1-Score** | 90.46% |

### Per-Class Insights
- **High performance** for *National Railway Museum*, *Royal Botanic Gardens*, and *Royal College of Physicians*, all achieving F1-scores above 91%.  
- **Slight weakness** in *Shakespeare Birthplace Trust* (Recall = 76.9%), likely due to limited representation and overlapping language with other classes.  

### Deployment
A Hugging Face pipeline was implemented for easy inference on unseen data:
```python
from transformers import pipeline
classifier = pipeline("text-classification", model="models/distilbert_best")
```

