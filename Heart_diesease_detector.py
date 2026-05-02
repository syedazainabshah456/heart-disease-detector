# ============================================
# HEART DISEASE DETECTOR - Complete ML Code
# AI Project - DUET
# ============================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import joblib

# ============================================
# STEP 1 - LOAD DATA
# ============================================

df = pd.read_csv('heart.csv')

print("=== RAW DATA ===")
print(df.head())
print("Shape:", df.shape)

# ============================================
# STEP 2 - CLEAN DATA
# ============================================

# Remove useless columns
df = df.drop(['id', 'dataset'], axis=1)

# Convert target to 0 and 1
df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

# Remove rows with missing values
df = df.dropna()

# Convert text columns to numbers using dummy encoding
df = pd.get_dummies(df, drop_first=True)

# X = input features, y = target
X = df.drop('num', axis=1)
y = df['num']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n=== AFTER CLEANING ===")
print("Total patients:", X_scaled.shape[0])
print("Total features:", X_scaled.shape[1])
print("Feature names:", list(X.columns))
print("With heart disease:", y.sum())
print("Without heart disease:", (y==0).sum())
print("[DONE] Data Cleaned!")

# ============================================
# STEP 3 - SPLIT DATA
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ============================================
# STEP 4 - TRAIN MODELS
# ============================================

print("\n=== TRAINING MODELS ===")

# Model 1 - Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))
print(f"Logistic Regression Accuracy: {lr_acc * 100:.2f}%")

# Model 2 - Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"Random Forest Accuracy: {rf_acc * 100:.2f}%")

# Model 3 - KNN
accuracies_knn = []
k_values = range(1, 20)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracies_knn.append(accuracy_score(y_test, knn.predict(X_test)))

best_k = list(k_values)[accuracies_knn.index(max(accuracies_knn))]
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
knn_acc = accuracy_score(y_test, knn_best.predict(X_test))
print(f"KNN Best K={best_k}, Accuracy: {knn_acc * 100:.2f}%")

print("[DONE] All Models Trained!")

# ============================================
# STEP 5 - CHARTS AND EVALUATION
# ============================================

print("\n=== GENERATING CHARTS ===")

# Chart 1 - Accuracy Comparison
plt.figure(figsize=(6, 4))
models = ['Logistic Regression', 'Random Forest', 'KNN']
accuracies = [lr_acc, rf_acc, knn_acc]
plt.bar(models, accuracies, color=['steelblue', 'seagreen', 'purple'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0.7, 1.0)
plt.tight_layout()
plt.savefig('accuracy_comparison.png')
plt.show()
print("Chart 1 saved!")

# Chart 2 - Confusion Matrix Logistic Regression
cm_lr = confusion_matrix(y_test, lr.predict(X_test))
ConfusionMatrixDisplay(cm_lr, display_labels=['No Disease', 'Disease']).plot()
plt.title("Logistic Regression - Confusion Matrix")
plt.savefig('confusion_matrix_lr.png')
plt.show()
print("Chart 2 saved!")

# Chart 3 - Confusion Matrix Random Forest
cm_rf = confusion_matrix(y_test, rf.predict(X_test))
ConfusionMatrixDisplay(cm_rf, display_labels=['No Disease', 'Disease']).plot()
plt.title("Random Forest - Confusion Matrix")
plt.savefig('confusion_matrix_rf.png')
plt.show()
print("Chart 3 saved!")

# Chart 4 - KNN K Value vs Accuracy
plt.figure(figsize=(8, 4))
plt.plot(k_values, accuracies_knn, marker='o', color='purple')
plt.title('KNN - Finding Best K Value')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig('knn_k_values.png')
plt.show()
print("Chart 4 saved!")

# Chart 5 - Feature Importance
feature_names = X.columns
importances = pd.Series(rf.feature_importances_, index=feature_names)
importances.sort_values().plot(kind='barh', color='coral', figsize=(8, 6))
plt.title('Which Factors Matter Most?')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
print("Chart 5 saved!")

# ============================================
# STEP 6 - SAVE MODEL + FEATURES
# ============================================

print("\n=== FINAL RESULTS ===")
print(f"Logistic Regression : {lr_acc * 100:.2f}%")
print(f"Random Forest       : {rf_acc * 100:.2f}%")
print(f"KNN (K={best_k})         : {knn_acc * 100:.2f}%")

# Save model, scaler, and feature column names
joblib.dump(rf, 'heart_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(list(X.columns), 'feature_columns.pkl')  # IMPORTANT - app.py ke liye

print("\n[DONE] heart_model.pkl saved!")
print("[DONE] scaler.pkl saved!")
print("[DONE] feature_columns.pkl saved!")
print("[DONE] Project Complete! Ready for UI!")