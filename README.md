# bike-sharing-demand-prediction

Assignment 2 - Breast Cancer Classification

## a) Problem statement
Build and compare multiple machine learning classification models on a single dataset and deploy an interactive Streamlit app that allows model selection, test-data upload, and model result visualization.

## b) Dataset description
- Dataset: Breast Cancer Wisconsin Diagnostic dataset
- Source file in repository: [data/breast_cancer_full.csv](data/breast_cancer_full.csv)
- Rows: 569
- Features: 30 numeric features
- Target column: diagnosis (benign / malignant)
- Submission test file: [test_data.csv](test_data.csv)

## c) Github repository link
- Repository: https://github.com/kunapulipravallika/bike-sharing-demand-prediction

## d) Models used and comparison table

Models implemented:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier (Gaussian)
5. Random Forest Classifier (Ensemble)

Metrics source: [model/metrics.json](model/metrics.json)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.8948 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| kNN | 0.9474 | 0.9835 | 0.9737 | 0.8810 | 0.9250 | 0.8872 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Random Forest (Ensemble) | 0.9737 | 0.9950 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

## Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong AUC and balanced precision-recall, with robust overall MCC. |
| Decision Tree | Interpretable but weakest overall among tested models on this split. |
| kNN | Strong baseline with good precision; slightly lower recall than Logistic Regression. |
| Naive Bayes | Perfect precision but lower recall, so more false negatives than top models. |
| Random Forest (Ensemble) | Best overall on accuracy, F1, and MCC, with near-top AUC. |

Overall winner for this dataset: Random Forest (Ensemble)

## Streamlit app features implemented
1. CSV upload option for test data.
2. Model selection dropdown.
3. Display of required evaluation metrics for selected model.
4. Confusion matrix and classification report display.
5. Predictions table and downloadable prediction CSV.

## Repository structure

- [app.py](app.py)
- [requirements.txt](requirements.txt)
- [README.md](README.md)
- [test_data.csv](test_data.csv)
- [model/train_models.py](model/train_models.py)
- model artifacts (*.pkl) and [model/metrics.json](model/metrics.json)

## Run locally
1. Install dependencies:
   pip install -r requirements.txt
2. Train models and regenerate artifacts:
   python model/train_models.py
3. Start Streamlit app:
   streamlit run app.py

## Submission notes
- Add your BITS Virtual Lab execution screenshot to your final PDF.
- Add your deployed Streamlit app URL in this README before final submission.
- Copy this README content into your submitted PDF in the required order.
