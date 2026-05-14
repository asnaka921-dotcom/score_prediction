import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ─────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────
data = pd.read_csv("student_scores.csv")
print("=" * 50)
print("DATASET PREVIEW")
print("=" * 50)
print(data.head())
print(f"\nDataset Shape: {data.shape}")
print(f"Missing Values:\n{data.isnull().sum()}")

# ─────────────────────────────────────────
# 2. FEATURES AND TARGET
# ─────────────────────────────────────────
X = data[['Hours']]
y = data['Scores']

# ─────────────────────────────────────────
# 3. SPLIT DATA
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────
# 4. DEFINE MODELS
# ─────────────────────────────────────────
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree":     DecisionTreeRegressor(random_state=42),
    "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42)
}

# ─────────────────────────────────────────
# 5. TRAIN, PREDICT & EVALUATE ALL MODELS
# ─────────────────────────────────────────
results = {}

print("\n" + "=" * 50)
print("MODEL EVALUATION RESULTS")
print("=" * 50)

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    results[name] = {
        "model":  model,
        "y_pred": y_pred,
        "MSE":    mse,
        "RMSE":   rmse,
        "MAE":    mae,
        "R2":     r2
    }

    print(f"\n📌 {name}")
    print(f"   MSE  : {mse:.4f}")
    print(f"   RMSE : {rmse:.4f}")
    print(f"   MAE  : {mae:.4f}")
    print(f"   R²   : {r2:.4f}")

# ─────────────────────────────────────────
# 6. BEST MODEL
# ─────────────────────────────────────────
best_model_name = max(results, key=lambda k: results[k]["R2"])
print("\n" + "=" * 50)
print(f"🏆 Best Model: {best_model_name} (R² = {results[best_model_name]['R2']:.4f})")
print("=" * 50)

# ─────────────────────────────────────────
# 7. VISUALIZATION
# ─────────────────────────────────────────
X_range = pd.DataFrame(
    np.linspace(X.min(), X.max(), 300), columns=['Hours']
)

colors = {
    "Linear Regression": "red",
    "Decision Tree":     "green",
    "Random Forest":     "orange"
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Student Scores Prediction - Model Comparison", fontsize=16, fontweight='bold')

for ax, (name, res) in zip(axes, results.items()):
    ax.scatter(X, y, color="blue", alpha=0.6, label="Actual Data", s=50)
    ax.plot(X_range, res["model"].predict(X_range),
            color=colors[name], linewidth=2, label=f"{name}")
    ax.set_title(f"{name}\nR² = {res['R2']:.4f}", fontsize=12)
    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Scores")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()

# ─────────────────────────────────────────
# 8. METRICS COMPARISON BAR CHART
# ─────────────────────────────────────────
metrics_df = pd.DataFrame({
    name: {"R²": res["R2"], "RMSE": res["RMSE"], "MAE": res["MAE"]}
    for name, res in results.items()
}).T

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle("Model Metrics Comparison", fontsize=14, fontweight='bold')

for ax, metric in zip(axes2, ["R²", "RMSE", "MAE"]):
    bars = ax.bar(metrics_df.index, metrics_df[metric],
                  color=["red", "green", "orange"], alpha=0.8, edgecolor='black')
    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.set_xticks(range(len(metrics_df.index)))
    ax.set_xticklabels(metrics_df.index, rotation=10, fontsize=9)
    for bar, val in zip(bars, metrics_df[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.3f}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("metrics_comparison.png", dpi=150)
plt.show()

# ─────────────────────────────────────────
# 9. USER INPUT PREDICTION
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("PREDICT YOUR SCORE")
print("=" * 50)

try:
    hours = float(input("Enter study hours to predict score: ")) # Allow user input
    print(f"\nPredictions for {hours} hours of study:")
    # Convert the single input hour into a DataFrame with the correct feature name
    hours_df = pd.DataFrame([[hours]], columns=['Hours'])
    for name, res in results.items():
        # Use the DataFrame for prediction to avoid the UserWarning
        pred = res["model"].predict(hours_df)[0]
        print(f"  {name:<22}: {pred:.2f}")
except ValueError:
    print("Invalid input. Please enter a numeric value.")
