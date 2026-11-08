# Calories Burnt Prediction

Small learning project. The notebook `calories_predict.ipynb` trains an XGBoost regressor.

Data: Exercise and calories data (exercise.csv, calories.csv), predicting calories burnt. The data files are not included in this repo, so you need to download them yourself.

## Benchmark script

`benchmark.py` compares HistGradientBoosting, random forest and ridge regression with 5-fold cross-validation.
It does not use the real dataset. It runs on synthetic data generated with numpy/scikit-learn (2000 samples, 7 features), so the numbers below only compare the models on that data and say nothing about the real dataset.

## How to run

```
pip install numpy scikit-learn jupyter
jupyter notebook
python benchmark.py
```

## Results (5-fold CV, mean)

| Model | r2 | rmse | mae |
|---|---|---|---|
| HistGradientBoosting | 0.9931 | 6.35 | 4.35 |
| Random Forest (200 Trees) | 0.9853 | 9.28 | 6.24 |
| Ridge Regression (Linear) | 0.9164 | 22.1 | 17.46 |

Full numbers are in `benchmarks/benchmark_results.json`.

## Files

- `calories_predict.ipynb`: notebook
- `benchmark.py`: benchmark script
- `benchmarks/benchmark_results.json`: results
- `assets/benchmark_chart.svg`: chart of the results
