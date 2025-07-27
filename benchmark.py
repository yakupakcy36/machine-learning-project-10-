#!/usr/bin/env python3
"""
Benchmark Suite: Metabolic Energy Expenditure & Calorie Predictor
Evaluates HistGradientBoosting Regressor, Random Forest Regressor, and Ridge Regression
on physiological exercise telemetry (Gender, Age, Height, Weight, Duration, Heart Rate,
Body Temperature) using 5-Fold Cross-Validation.
"""

import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def generate_metabolic_telemetry_cohort(n_samples: int = 2000, random_state: int = 42):
    """
    Synthesizes physiological exercise telemetry calibrated to the Keytel et al. (2005)
    cardiorespiratory energy expenditure equation and thermogenic dynamics.
    """
    rng = np.random.default_rng(random_state)

    gender_male = rng.binomial(1, 0.496, size=n_samples).astype(float)
    age = rng.integers(20, 79, size=n_samples).astype(float)
    height_cm = np.where(
        gender_male == 1,
        rng.normal(178.5, 7.2, size=n_samples),
        rng.normal(164.8, 6.5, size=n_samples),
    )
    weight_kg = np.where(
        gender_male == 1,
        rng.normal(82.0, 11.0, size=n_samples),
        rng.normal(63.5, 9.5, size=n_samples),
    )
    duration_min = rng.uniform(2.0, 30.0, size=n_samples)
    heart_rate_bpm = np.clip(
        78.0 + 1.05 * duration_min + rng.normal(0.0, 6.8, size=n_samples),
        67.0,
        128.0,
    )
    body_temp_c = np.clip(
        37.1 + 0.095 * duration_min + 0.008 * (heart_rate_bpm - 80.0) + rng.normal(0.0, 0.18, size=n_samples),
        36.8,
        41.5,
    )

    # Keytel physiological expenditure formula (kcal/min * duration) with non-linear thermal coupling
    rate_male = (-55.0969 + 0.6309 * heart_rate_bpm + 0.1988 * weight_kg + 0.2017 * age) / 4.184
    rate_female = (-20.4022 + 0.4472 * heart_rate_bpm - 0.1263 * weight_kg + 0.074 * age) / 4.184
    kcal_per_min = np.where(gender_male == 1, rate_male, rate_female)
    thermal_factor = 1.0 + 0.035 * np.maximum(0.0, body_temp_c - 38.5) ** 2
    calories_kcal = np.clip(
        kcal_per_min * duration_min * thermal_factor + rng.normal(0.0, 2.1, size=n_samples),
        1.0,
        315.0,
    )

    X = np.column_stack([
        gender_male,
        age,
        height_cm,
        weight_kg,
        duration_min,
        heart_rate_bpm,
        body_temp_c,
    ])
    feature_names = [
        "Gender_Male",
        "Age_Years",
        "Height_cm",
        "Weight_kg",
        "Exercise_Duration_min",
        "Heart_Rate_bpm",
        "Body_Temp_Celsius",
    ]
    return X, calories_kcal, feature_names


def render_svg_chart(results: list, output_path: Path):
    """Generates a dark-mode vector performance chart for metabolic calorie prediction."""
    width, height = 860, 420
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        '  <defs>',
        '    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">',
        '      <stop offset="0%" stop-color="#0f172a"/>',
        '      <stop offset="100%" stop-color="#1e293b"/>',
        '    </linearGradient>',
        '  </defs>',
        f'  <rect width="{width}" height="{height}" rx="12" fill="url(#bgGrad)" stroke="#334155" stroke-width="1.5"/>',
        '  <text x="32" y="42" fill="#f8fafc" font-family="Segoe UI, Inter, sans-serif" font-size="20" font-weight="700">'
        'Metabolic Energy Expenditure &amp; Calorie Predictor — 5-Fold CV</text>',
        '  <text x="32" y="66" fill="#94a3b8" font-family="Segoe UI, Inter, sans-serif" font-size="13">'
        'Physiological Telemetry Cohort (N = 2,000 Sessions, d = 7 Biometrics) | Keytel Cardiorespiratory Kinetics</text>',
        '  <rect x="32" y="84" width="12" height="12" rx="3" fill="#38bdf8"/>',
        '  <text x="50" y="94" fill="#cbd5e1" font-family="Segoe UI, Inter, sans-serif" font-size="12" font-weight="600">R² Score (Higher is Better)</text>',
        '  <rect x="240" y="84" width="12" height="12" rx="3" fill="#34d399"/>',
        '  <text x="258" y="94" fill="#cbd5e1" font-family="Segoe UI, Inter, sans-serif" font-size="12" font-weight="600">RMSE (kcal — Lower is Better)</text>',
        '  <rect x="465" y="84" width="12" height="12" rx="3" fill="#f97316"/>',
        '  <text x="483" y="94" fill="#cbd5e1" font-family="Segoe UI, Inter, sans-serif" font-size="12" font-weight="600">MAE (kcal — Lower is Better)</text>',
    ]

    chart_left = 250
    chart_right = 785
    chart_width = chart_right - chart_left
    top_y = 124
    group_height = 90

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        gx = chart_left + tick * chart_width
        svg.append(
            f'  <line x1="{gx:.1f}" y1="{top_y - 8}" x2="{gx:.1f}" y2="{top_y + 3 * group_height - 12}" '
            f'stroke="#334155" stroke-dasharray="4,4" stroke-width="1"/>'
        )
        svg.append(
            f'  <text x="{gx:.1f}" y="{top_y + 3 * group_height + 10}" fill="#64748b" '
            f'font-family="Segoe UI, Inter, sans-serif" font-size="11" text-anchor="middle">{tick:.2f}</text>'
        )

    for idx, entry in enumerate(results):
        gy = top_y + idx * group_height
        model_name = entry["model"]
        svg.append(
            f'  <text x="{chart_left - 14}" y="{gy + 32}" fill="#f1f5f9" '
            f'font-family="Segoe UI, Inter, sans-serif" font-size="13" font-weight="600" text-anchor="end">{model_name}</text>'
        )
        r2_w = max(2.0, entry["r2"] * chart_width)
        svg.append(f'  <rect x="{chart_left}" y="{gy}" width="{r2_w:.1f}" height="16" rx="3" fill="#38bdf8"/>')
        svg.append(
            f'  <text x="{chart_left + r2_w + 8:.1f}" y="{gy + 12}" fill="#e2e8f0" '
            f'font-family="Consolas, monospace" font-size="11" font-weight="600">R² = {entry["r2"]:.4f}</text>'
        )

        rmse_w = max(2.0, min(1.0, entry["rmse"] / 15.0) * chart_width)
        svg.append(f'  <rect x="{chart_left}" y="{gy + 21}" width="{rmse_w:.1f}" height="16" rx="3" fill="#34d399"/>')
        svg.append(
            f'  <text x="{chart_left + rmse_w + 8:.1f}" y="{gy + 33}" fill="#e2e8f0" '
            f'font-family="Consolas, monospace" font-size="11" font-weight="600">RMSE = {entry["rmse"]:.2f} kcal</text>'
        )

        mae_w = max(2.0, min(1.0, entry["mae"] / 15.0) * chart_width)
        svg.append(f'  <rect x="{chart_left}" y="{gy + 42}" width="{mae_w:.1f}" height="16" rx="3" fill="#f97316"/>')
        svg.append(
            f'  <text x="{chart_left + mae_w + 8:.1f}" y="{gy + 54}" fill="#e2e8f0" '
            f'font-family="Consolas, monospace" font-size="11" font-weight="600">MAE = {entry["mae"]:.2f} kcal</text>'
        )

    svg.append('</svg>')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main():
    X, y, feature_names = generate_metabolic_telemetry_cohort()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.06, max_depth=6, random_state=42
        ),
        "Random Forest (200 Trees)": RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=3, random_state=42
        ),
        "Ridge Regression (Linear)": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0, random_state=42)),
        ]),
    }

    benchmark_rows = []
    for name, model in models.items():
        oof_preds = np.zeros(len(y))
        for train_idx, val_idx in cv.split(X):
            model.fit(X[train_idx], y[train_idx])
            oof_preds[val_idx] = model.predict(X[val_idx])

        rmse = float(np.sqrt(mean_squared_error(y, oof_preds)))
        mae = float(mean_absolute_error(y, oof_preds))
        r2 = float(r2_score(y, oof_preds))

        benchmark_rows.append({
            "model": name,
            "r2": round(r2, 4),
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
        })

    benchmark_rows.sort(key=lambda r: r["r2"], reverse=True)

    payload = {
        "project": "Metabolic Energy Expenditure & Calorie Predictor",
        "dataset": "Physiological Exercise Telemetry Cohort (N=2,000, d=7)",
        "validation_scheme": "5-Fold Cross-Validation",
        "features": feature_names,
        "results": benchmark_rows,
    }

    base_dir = Path(__file__).resolve().parent
    json_path = base_dir / "benchmarks" / "benchmark_results.json"
    svg_path = base_dir / "assets" / "benchmark_chart.svg"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    render_svg_chart(benchmark_rows, svg_path)
    print(f"Saved benchmark results to {json_path}")
    print(f"Saved benchmark chart to {svg_path}")


if __name__ == "__main__":
    main()
