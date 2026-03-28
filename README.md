# UEDU - Mental Health Detection in Student Writing

> Open-source NLP toolkit for detecting mental health distress signals in student writing using psycholinguistic features. 100% local, privacy-first, no cloud AI needed.

**Understand. Evaluate. Describe. Uplift.**

UEDU is an open-source Python library for mental health signal detection in student writing. It extracts 40 psycholinguistic features (emotion, cognition, social dynamics, linguistic patterns) and uses a pre-trained XGBoost classifier to identify distress signals -- with a deterministic, explainable "Glass Box" report engine grounded in MHFA ALGEE and BC ERASE clinical protocols.

All processing runs locally. Zero data leaves the machine. Designed for educators, school counselors, and mental health researchers worldwide. Originally developed and tested in British Columbia, Canada (FOIPPA-compliant).

**Keywords**: mental health NLP, student wellbeing, psycholinguistic analysis, suicide detection, text classification, XGBoost, explainable AI, school counseling, SHAP, feature engineering, distress detection, education technology

## Why This Exists

Teachers read hundreds of student assignments. Some contain distress signals -- hopelessness, self-harm language, social withdrawal -- buried in ordinary schoolwork. These signals are easy to miss at scale, and existing tools either require cloud processing (violating student privacy laws) or lack clinical grounding.

UEDU solves this: a fully local, deterministic pipeline that flags concerning writing patterns and gives teachers actionable, protocol-backed guidance.

## Quick Start

```bash
git clone https://github.com/harold-wang-dev/uedu.git
cd uedu
pip install -r requirements.txt
```

```python
import pickle
import numpy as np
from src.features.registry import extract_all_features, ALL_FEATURE_NAMES

# 1. Extract 40 psycholinguistic features
text = "I feel completely worthless. Nothing matters anymore."
features = extract_all_features(text, use_spacy=False)

# 2. Load pre-trained model and predict
with open("models/M5.pkl", "rb") as f:
    model_data = pickle.load(f)

X = np.array([[features.get(name, 0.0) for name in ALL_FEATURE_NAMES]])
prob = float(model_data["classifier"].predict_proba(X)[0, 1])

print(f"Distress probability: {prob:.1%}")
# Distress probability: 82.3%
```

Pre-trained models (M5.pkl, M6.pkl) are included. No dataset download required for inference.

## How It Works

```
Student Text
     |
     v
[UNDERSTAND] 40 Psycholinguistic Features (deterministic)
     |           6 groups: Affective, Cognitive, Temporal,
     |           Structural, Social, Linguistic Process
     v
[EVALUATE] XGBoost Classifier (local, pre-trained)
     |           M5: 40 features, 87.3% accuracy
     |           Trained on 232K posts, static model file
     v
[DESCRIBE] Glass Box Report Engine (deterministic, no AI)
     |           Risk level: None / Mild / High
     |           Top-3 features with clinical interpretations
     |           MHFA ALGEE + BC ERASE protocol mapping
     v
[UPLIFT] Teacher Guidance
               Actionable steps for each risk level
               Coping strategies, referral pathways
```

Every step is deterministic. Same input always produces the same output. No network calls.

## Installation

**Prerequisites**: Python 3.10+, pip

```bash
git clone https://github.com/harold-wang-dev/uedu.git
cd uedu
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt_tab')"
```

spaCy is optional. Pass `use_spacy=False` to skip POS-based temporal features (faster, slightly less accurate).

## The 40 Features

| Group | Features | Examples |
|-------|----------|---------|
| **Affective** (A1--A8) | 8 | Negative emotion density, death/crisis words, anger, sadness, anxiety |
| **Cognitive** (C1--C7) | 7 | Cognitive distortions, certainty, tentative language, negation |
| **Temporal** (T1--T6) | 6 | Past/present/future tense ratios, urgency score |
| **Structural** (S1--S8) | 8 | Sentence length, lexical diversity, readability, word count |
| **Social** (SO1--SO6) | 6 | First-person singular, self-vs-other ratio, social references |
| **Linguistic Process** (LP1--LP5) | 5 | Function words, articles, health/body words |

All features are deterministic word-ratio calculations. No randomness, no model inference.

## Model Performance

Trained and evaluated on 231,938 Reddit posts (5-fold cross-validation):

| Model | Features | Accuracy | F1 | AUC |
|-------|----------|:--------:|:--:|:---:|
| M1 | TF-IDF (256) + LogReg | 88.92% | 0.887 | 0.951 |
| M2 | Psycholinguistic (40) + LogReg | 83.50% | 0.836 | 0.906 |
| M5 | Psycholinguistic (40) + XGBoost | 87.26% | 0.870 | 0.941 |
| M6 | TF-IDF + Psycholinguistic (296) + XGBoost | 90.47% | 0.903 | 0.964 |
| M7 | LLM Features (8) + XGBoost | 93.47% | 0.935 | 0.978 |
| M9 | All Features (304) + XGBoost | **94.66%** | **0.947** | **0.986** |

M5 is the recommended model for local deployment (40 features, no TF-IDF vocabulary dependency).

## Fairness

We audited the model on 24,728 ASAP student essays across ELL status, race/ethnicity, and gender:

- **ESL/ELL students**: False-positive rate = 1.7% (M5), **lower** than non-ELL at 2.4% (ratio 0.72x, p=0.022)
- **Hypothesis rejected**: The model measures distress signals, not writing quality deficits

The model does not penalize students for being ESL learners. See `src/fairness/metrics.py` for the full audit methodology.

## Project Structure

```
uedu/
  models/
    M5.pkl                  # Pre-trained: 40 psycholinguistic features + XGBoost
    M6.pkl                  # Pre-trained: TF-IDF + psycholinguistic + XGBoost
  src/
    features/
      registry.py           # Feature orchestrator (extract_all_features)
      affective.py          # A1-A8: emotion, anger, death words
      cognitive.py          # C1-C7: distortions, certainty, negation
      temporal.py           # T1-T6: tense ratios, urgency
      structural.py         # S1-S8: sentence length, readability
      social.py             # SO1-SO6: pronouns, self-focus
      linguistic_process.py # LP1-LP5: function words, health terms
    data/
      preprocessor.py       # Text cleaning (URLs, mentions, whitespace)
      loader.py             # Dataset loading (Kaggle, Zenodo)
      splitter.py           # Train/val/test splitting
    models/
      trainer.py            # M1-M9 training pipeline
      evaluator.py          # Cross-validation and metrics
    fairness/
      metrics.py            # Demographic parity, equalized odds, chi-squared FPR
  results/
    exp1/                   # Model comparison results
    exp4/                   # SHAP feature importance rankings
  tests/
    test_features.py        # Feature extraction unit tests
```

## Datasets

Training data is **not included** due to privacy and licensing. Download separately if you want to retrain:

| Dataset | Size | Purpose | Download |
|---------|------|---------|----------|
| Kaggle Suicide Detection | 232K posts | Primary training data | `kaggle datasets download -d nikhileswarkomati/suicide-watch` |
| ASAP 2.0 Student Essays | 24.7K essays | Fairness audit | `kaggle datasets download -d lburleigh/asap-2-0` |
| Zenodo Reddit Mental Health | Multiple subreddits | Extended training | [zenodo.org/record/3941387](https://zenodo.org/record/3941387) |

You do **not** need any dataset to run inference. The pre-trained M5 and M6 models are included.

## Training Your Own Model

```bash
# 1. Download the Kaggle dataset
kaggle datasets download -d nikhileswarkomati/suicide-watch \
  -p data/raw/kaggle/suicide-depression/

# 2. Unzip
unzip data/raw/kaggle/suicide-depression/suicide-watch.zip \
  -d data/raw/kaggle/suicide-depression/

# 3. Train M5 (psycholinguistic features + XGBoost)
python -m src.models.trainer --model M5
```

## Running Tests

```bash
pytest tests/ -v
```

## Key References

- Al-Mosaiwi & Johnstone (2018). Absolutist thinking in suicidal ideation. *Clinical Psychological Science*.
- Edwards & Holtzman (2017). Self-referential language and depression. *Journal of Language and Social Psychology*.
- Pennebaker (2011). *The Secret Life of Pronouns*. Bloomsbury Press.
- Mental Health First Aid (MHFA) Canada -- ALGEE action plan.
- BC ERASE: [gov.bc.ca/erase](https://www2.gov.bc.ca/gov/content/erase).

## Links

- **Website**: [uedu.ca](https://uedu.ca)
- **Live Demo**: [uedu.ca/demo](https://uedu.ca/demo)

## Citation

```bibtex
@software{uedu2026,
  title   = {UEDU: Psycholinguistic Mental Health Signal Detection in Student Writing},
  author  = {{UEDU Team}},
  year    = {2026},
  url     = {https://github.com/harold-wang-dev/uedu},
  note    = {Understand, Evaluate, Describe, Uplift}
}
```

## About

UEDU was developed by Harold W., a Grade 11 student in British Columbia, Canada, as a Health Sciences research project for the Greater Vancouver Regional Science Fair (GVRSF 2026).

The goal is to give educators a transparent, privacy-respecting tool for early identification of student mental health concerns, so that no student's silent distress goes unnoticed.

## License

MIT License. See [LICENSE](LICENSE) for details.
