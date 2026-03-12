# Check-Worthiness Detection for Spanish Political Claims

This repository contains the experimental code and artifacts for automatic **check-worthiness classification** in Spanish text (mainly tweets).  
The project was developed as the implementation base for:

- an undergraduate Systems Engineering final thesis (UNICEN), and
- a research paper co-authored with **Antonela Tommasel**.

The main objective is to prioritize which claims should be fact-checked first when human verification resources are limited.

## Academic context

- **Thesis (UNICEN):** _Trabajo final - Galo Pianciola_ (Systems Engineering).
- Thesis full text: [UNICEN repository (RIDAA)](https://ridaa.unicen.edu.ar:8443/server/api/core/bitstreams/7765374e-5d10-456a-bff0-6ef7822fd9de/content)
- **Paper (CLEI 2024):**  
  _Hacia el fact-checking automático: Un estudio exploratorio sobre la identificación de frases críticas para verificación_  
  Authors: **Galo Emanuel Pianciola Bartol**, **Antonela Tommasel**.
- Paper publication: [IEEE Xplore](https://ieeexplore.ieee.org/document/10700241)

## Repository structure

| Path | Description |
| --- | --- |
| `main.ipynb` | Main notebook with full pipeline: loading, preprocessing, feature extraction, modeling, evaluation, plots. |
| `data/` | Raw CLEF datasets, merged train/dev/test files, and saved experiment outputs. |
| `requirements.txt` | Core Python dependencies. |

## Data

The notebook merges Spanish-language check-worthiness datasets from:

- CLEF CheckThat! 2021
- CLEF CheckThat! 2022
- CLEF CheckThat! 2023

It harmonizes schema/labels and builds unified splits:

| File | Rows | Check-worthy label ratio (`class_label=1`) |
| --- | ---: | ---: |
| `data/train.csv` | 7,491 | 19.46% |
| `data/dev.csv` | 5,435 | 13.12% |
| `data/test.csv` | 7,976 | 14.15% |
| `data/complete.csv` | 14,991 | 16.46% |

## Methodology implemented

The notebook evaluates multiple families of approaches:

- **Preprocessing (spaCy Spanish):**
  - lemmatization,
  - stopword/non-alpha filtering,
  - mention and URL masking (`USER`, `URL`),
  - hashtag handling (including UpperCamelCase splitting).
- **Text representations:**
  - TF-IDF,
  - Word2Vec,
  - FastText,
  - InferSent,
  - BETO embeddings (`dccuchile/bert-base-spanish-wwm-uncased`),
  - BETO fine-tuning (sequence classification).
- **Classifiers:**
  - SVM,
  - Naive Bayes,
  - Random Forest,
  - Logistic Regression,
  - ensemble methods (voting, bagging, stacking, AdaBoost, Gradient Boosting).
- **LLM baselines:**
  - GPT,
  - Gemma,
  - Llama,
  - Mistral (few-shot / zero-shot experiment blocks).
- **Evaluation:**
  - Precision, Recall, F1, Weighted F1, Accuracy, Balanced Accuracy, MCC,
  - 10-fold cross-validation (`data/final_results_cv.json`),
  - paired statistical tests (t-test / Wilcoxon, depending on normality).

## Results snapshot

Using `data/final_results.json` (test-set metrics), top models by F1:

| Model | F1 | Balanced Acc. | MCC |
| --- | ---: | ---: | ---: |
| `ensemble_adaboost` | 0.5795 | 0.7999 | 0.5089 |
| `beto_fine_tuning` | 0.5661 | 0.7658 | 0.4895 |
| `bayes_beto` | 0.5261 | 0.7779 | 0.4478 |
| `logistic_regression_beto` | 0.4995 | 0.6888 | 0.4440 |

From `data/final_results_cv.json` (10-fold CV means), top by F1 mean:

| Model | CV F1 mean | CV F1 std |
| --- | ---: | ---: |
| `ensemble_adaboost` | 0.6114 | 0.0830 |
| `beto_fine_tuning` | 0.6068 | 0.0734 |
| `ensemble_stacking` | 0.5554 | 0.0575 |
| `logistic_regression_beto` | 0.5392 | 0.0554 |

Interpretation at high level: BETO-based and ensemble approaches are the strongest in this repository’s experiments, while LLM zero/few-shot blocks show high recall in some settings but weak precision/F1 trade-offs.

## How to run

### 1) Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download es_core_news_lg
```

### 2) Optional/external components (needed for full notebook coverage)

- **OpenAI API**: set `OPENAI_API_KEY` in `.env` for GPT cells.
- **FastText**: `fasttext` is commented in `requirements.txt`; install manually if you want those pipelines.
- **InferSent**: notebook imports `InferSent.models`, so you need the InferSent code available in your environment.
- **Transformer models** (BETO) are downloaded on first use.

### 3) Execute notebook

```bash
jupyter lab main.ipynb
```

Notes:

- The notebook includes a Colab-specific section (`google.colab` drive mounting). Skip/adapt it for local runs.
- Precomputed artifacts already exist in `data/` (`final_results.json`, `final_results_cv.json`, merged datasets).

## Reproducibility notes

- This is a research notebook repository (not yet packaged as a Python module).
- Some exploratory cells depend on files not currently tracked in `data/` (for example, certain LLM prediction JSON files used by specific sections).
- `data/final_results.csv` and `data/final_results.json` are not fully equivalent (different model coverage / naming); prefer JSON files as the most complete experiment summary.

## Citation

If you use this repository, please cite the CLEI paper and reference the thesis work.

```bibtex
@inproceedings{pianciola2024checkworthiness,
  title     = {Hacia el fact-checking autom{\\'a}tico: Un estudio exploratorio sobre la identificaci{\\'o}n de frases cr{\\'i}ticas para verificaci{\\'o}n},
  author    = {Pianciola Bartol, Galo Emanuel and Tommasel, Antonela},
  year      = {2024},
  booktitle = {CLEI}
}
```

## Acknowledgment

This work was developed in an academic context at UNICEN, with research collaboration and guidance from **Antonela Tommasel**.
