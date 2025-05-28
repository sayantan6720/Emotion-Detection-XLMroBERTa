
---

## Setup & Installation

1. **Clone the repository** and navigate to the project directory.

2. **Install dependencies** (preferably in a virtual environment):

   ```bash
   pip install pandas torch transformers peft tqdm numpy datasets scikit-learn sentencepiece modal
   ```

   - For Jupyter support, you may also need:
     ```bash
     pip install ipywidgets
     ```

3. **(Optional) Install `hf_xet` for faster HuggingFace downloads:**
   ```bash
   pip install huggingface_hub[hf_xet]
   ```

---

## Data

- The project uses the [XED dataset](https://github.com/Helsinki-NLP/XED/tree/master), which contains multilingual text samples (movie subtitles) labeled with one or more emotions.
- The dataset uses Plutchik's 8 core emotions and is multi-label.
- Data files (`train.tsv` and `test.tsv`) should be placed in the `datasets/` directory.
- The XED dataset is licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

## Exploratory Data Analysis (EDA)

- Open `XED_EDA.ipynb` to explore the dataset.
- The notebook provides insights into:
  - Distribution of emotions
  - Language coverage
  - Label frequencies
  - Any preprocessing or cleaning steps

---

## Model Training & Evaluation

The main training pipeline is in `xlm_finetune.ipynb` and includes:

- **Model:** XLM-RoBERTa (base) for sequence classification, adapted for multi-label emotion detection.
- **Label Processing:** Emotions are one-hot encoded for multi-label classification.
- **Data Splitting:** Stratified by language, with 85% for training and 15% for validation.
- **Fine-tuning:** Uses the PEFT (Parameter-Efficient Fine-Tuning) library with LoRA (Low-Rank Adaptation) for efficient training.
- **Trainer:** HuggingFace `Trainer` API with custom data collator for multi-label tasks.
- **Evaluation:** Computes micro, macro, and weighted F1 scores overall and per language.

### Running the Training

The notebook uses the `modal` library to run training in a containerized environment (with GPU support if available):

```python
with modal.enable_output():
    with app.run():
        df_train = tsv_to_df.remote("/root/train.tsv")
        df_test = tsv_to_df.remote("/root/test.tsv")
        print(train_and_evaluate.remote(df_train, df_test))
```

---

## Results

- The model outputs F1 scores (micro, macro, weighted) both overall and for each language.
- Example (from a sample run):

  ```
  ===== OVERALL F1 SCORES =====
  Micro F1: 0.0377
  Macro F1: 0.0336
  Weighted F1: 0.0335

  ===== F1 SCORES BY LANGUAGE =====
  Language: Greek (Examples: 1617)
    Micro F1: 0.0336
    Macro F1: 0.0309
    Weighted F1: 0.0293
  ...
  ```

---

## References

- [XED: Cross-lingual Emotion Dataset (Helsinki-NLP/XED)](https://github.com/Helsinki-NLP/XED/tree/master)
- [XLM-RoBERTa: Cross-lingual Language Model](https://arxiv.org/abs/1911.02116)
- [PEFT: Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [Modal: Cloud Functions for ML](https://modal.com/)

### Related Publications

- Öhman, E., Kajava, K., Tiedemann, J. and Honkela, T., 2018, October. Creating a dataset for multilingual fine-grained emotion-detection using gamification-based annotation. In Proceedings of the 9th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis (pp. 24-30).
- Öhman, E.S. and Kajava, K.S., 2018. Sentimentator: Gamifying fine-grained sentiment annotation. Digital Humanities in the Nordic Countries 2018.
- Kajava, K.S., Öhman, E.S., Hui, P. and Tiedemann, J., 2020. Emotion Preservation in Translation: Evaluating Datasets for Annotation Projection. In Digital Humanities in the Nordic Countries 2020. CEUR Workshop Proceedings.
- Öhman, E., 2020. Challenges in Annotation: Annotator Experiences from a Crowdsourced Emotion Annotation Task. In Digital Humanities in the Nordic Countries 2020. CEUR Workshop Proceedings.

---

## Notes

- For best performance, ensure your kernel version is >= 5.5.0 if running in a Linux environment.
- The code is designed for multi-label emotion detection and can be adapted for other multi-label text classification tasks.

---
