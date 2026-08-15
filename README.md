# Bike Sharing Demand Prediction

## Problem statement
Build and compare multiple machine learning models for bike rental demand prediction and identify the best-performing model using cross-validation metrics.

## Dataset description
- Training dataset: [data/bike_train_full.csv](data/bike_train_full.csv)
- Test dataset: [test_data.csv](test_data.csv)
- Training rows: 10,450
- Test rows: 2,613
- Input features include weather, seasonality, and time-based variables.
- Target variable: count (bike rental count)

Training columns:
- datetime, season, holiday, workingday, weather, temp, atemp, humidity, windspeed, casual, registered, count

Test columns:
- datetime, season, holiday, workingday, weather, temp, atemp, humidity, windspeed

## GitHub repository link
- https://github.com/kunapulipravallika

## Live Streamlit app link
- Not available currently (no Streamlit account).

## Model comparison table with metrics

Source metrics file: [models/metrics.json](models/metrics.json)

| Model | CV RMSLE (mean) | CV RMSLE (std) | CV MAE (mean) | Artifact |
|---|---:|---:|---:|---|
| gradient_boosting | 0.35910 | 0.00778 | 24.23389 | [models/gradient_boosting.pkl](models/gradient_boosting.pkl) |
| random_forest | 0.37809 | 0.00949 | 24.23435 | [models/random_forest.pkl](models/random_forest.pkl) |
| knn | 0.59418 | 0.01069 | 45.73926 | [models/knn.pkl](models/knn.pkl) |
| linear_regression | 0.69409 | 0.01244 | 68.43465 | [models/linear_regression.pkl](models/linear_regression.pkl) |
| ridge | 0.69409 | 0.01242 | 68.42867 | [models/ridge.pkl](models/ridge.pkl) |
| lasso | 0.69433 | 0.01229 | 68.37216 | [models/lasso.pkl](models/lasso.pkl) |

## Observations and overall winner
- Tree-based ensembles clearly outperform linear and neighbor-based baselines for this dataset.
- Gradient Boosting achieves the lowest CV RMSLE (0.35910), making it the overall winner.
- Random Forest is a strong second choice with very similar MAE and slightly higher RMSLE.
- Linear, Ridge, and Lasso underfit relative to ensemble methods, likely due to non-linear demand patterns.

Overall winner: gradient_boosting

## Required plots and tables

The required visual outputs are included:
- [q2_eda_plots.png](q2_eda_plots.png)
- [q2_hourly_pattern.png](q2_hourly_pattern.png)
- [q2_correlation_heatmap.png](q2_correlation_heatmap.png)
- [q8_residual_plots.png](q8_residual_plots.png)

Preview in markdown:

![EDA Plots](q2_eda_plots.png)

![Hourly Pattern](q2_hourly_pattern.png)

![Correlation Heatmap](q2_correlation_heatmap.png)

![Residual Plots](q8_residual_plots.png)

## Reproduce results
1. Install dependencies:
   pip install -r requirements.txt
2. Train models:
   /Library/Developer/CommandLineTools/usr/bin/python3 models/train_models.py
3. Launch Streamlit app:
   streamlit run app.py

## Submission checklist
- Add your Name and BITS ID in the final document.
- Insert clickable GitHub and Streamlit links.
- Add BITS Virtual Lab screenshot in the final PDF.
- Ensure this README content is copied into section 4 of your submission format.
