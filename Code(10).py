import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

# ✅ Drop Unnecessary Columns (If Needed)
df.drop(columns=['Person ID', 'Occupation', 'BMI Category', 'Blood Pressure','Daily Steps','Physical Activity Level'], inplace=True, errors='ignore')

# ✅ Convert Categorical Columns to Numeric
df = pd.get_dummies(df, columns=['Gender', 'Sleep Disorder'], drop_first=True)

# ✅ Define Features (X) and Target (y)
X = df.drop(columns=['Quality of Sleep'])  # Inputs
y = df['Quality of Sleep']  # Target Variable

df.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train Random Forest Model (Baseline)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# ✅ Predictions
y_pred = rf_model.predict(X_test)

# ✅ Evaluate Baseline Model
print("\n🎯 Baseline Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": rf_model.feature_importances_})
feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=feature_importance["Importance"], y=feature_importance["Feature"], palette="coolwarm")
plt.title("📌 Feature Importance (Baseline Model)")
plt.show()


# ✅ Train Optimized Model
rf_model_tuned = RandomForestClassifier(
    n_estimators=200,  # More trees for better accuracy
    max_depth=10,      # Deeper trees
    min_samples_split=5,  # More robust splits
    random_state=42
)

rf_model_tuned.fit(X_train, y_train)
y_pred_tuned = rf_model_tuned.predict(X_test)

# ✅ Evaluate Tuned Model
print("\n✅ Tuned Model Accuracy:", accuracy_score(y_test, y_pred_tuned))
print("\n📊 Classification Report (Tuned):\n", classification_report(y_test, y_pred_tuned))

# ✅ Feature Importance Plot (Tuned)
feature_importance_tuned = pd.DataFrame({"Feature": X.columns, "Importance": rf_model_tuned.feature_importances_})
feature_importance_tuned = feature_importance_tuned.sort_values(by="Importance", ascending=False)
