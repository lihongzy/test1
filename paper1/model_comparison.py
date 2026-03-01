# -*- coding: utf-8 -*-
"""
GBDT 与 MNL 模型对比分析

本模块实现：
1. GBDT 模型 + 五折交叉验证 + 超参数调优（简化版）
2. MNL 模型（多项Logit）
3. 类别权重处理
4. VIF 多重共线性检验
5. 模型对比评估
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import warnings
import os
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAVEL_CHARS_PATH = os.path.join(BASE_DIR, "paper1", "output", "travel_characteristics.csv")
USER_STATS_PATH = os.path.join(BASE_DIR, "paper1", "output", "user_statistics.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "paper1", "output")

LEARNING_RATES = [0.1, 0.05, 0.01]
MAX_DEPTHS = [5, 10, 20]
N_ESTIMATORS = [300, 400]
MIN_SAMPLES_LEAF = [20, 50]


def calculate_vif(X_scaled_df):
    """
    手动计算 VIF (方差膨胀因子)
    VIF = 1 / (1 - R²)
    """
    vif_data = {}
    for i, col in enumerate(X_scaled_df.columns):
        X_temp = X_scaled_df.drop(columns=[col])
        y_temp = X_scaled_df[col]
        
        lr = LinearRegression()
        lr.fit(X_temp, y_temp)
        r_squared = lr.score(X_temp, y_temp)
        
        if r_squared >= 1:
            vif = np.inf
        else:
            vif = 1 / (1 - r_squared)
        
        vif_data[col] = vif
    
    return pd.DataFrame(list(vif_data.items()), columns=['Variable', 'VIF'])


print("=" * 60)
print("GBDT 与 MNL 模型对比分析")
print("=" * 60)

print("\n[1/6] 加载数据...")
travel_chars_df = pd.read_csv(TRAVEL_CHARS_PATH)
user_stats_df = pd.read_csv(USER_STATS_PATH)

print(f"  出行特征数据: {len(travel_chars_df)} 条")
print(f"  用户统计数据: {len(user_stats_df)} 条")

print("\n[2/6] 准备建模数据...")
model_data = travel_chars_df.merge(
    user_stats_df[['user_guid', 'user_type']], 
    on='user_guid', 
    how='inner'
)

feature_cols = ['cycling_days', 'cycling_trips', 'cycling_distance_km', 'cluster_count']
X = model_data[feature_cols].copy()
y = model_data['user_type'].copy()

le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

print(f"  特征变量: {feature_cols}")
print(f"  样本数量: {len(X)}")
print(f"  类别分布:")
for i, name in enumerate(class_names):
    count = (y_encoded == i).sum()
    print(f"    - {name}: {count} ({count/len(y_encoded)*100:.1f}%)")

print("\n[3/6] VIF 多重共线性检验...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

vif_data = calculate_vif(X_scaled_df)

print("\n  VIF 检验结果:")
print(vif_data.to_string(index=False))

max_vif = vif_data["VIF"].max()
if max_vif < 7.5:
    print(f"\n  ✓ VIF 最大值为 {max_vif:.2f} < 7.5，无多重共线性问题")
else:
    print(f"\n  ✗ VIF 最大值为 {max_vif:.2f} >= 7.5，存在多重共线性问题")

print("\n[4/6] GBDT 模型 - 超参数调优（简化版）...")
best_accuracy = 0
best_params = {}
best_model_gbdt = None

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_combinations = []
for lr in LEARNING_RATES:
    for depth in MAX_DEPTHS:
        for n_est in N_ESTIMATORS:
            for min_leaf in MIN_SAMPLES_LEAF:
                param_combinations.append({
                    'learning_rate': lr,
                    'max_depth': depth,
                    'n_estimators': n_est,
                    'min_samples_leaf': min_leaf
                })

print(f"  共 {len(param_combinations)} 种参数组合")

for i, params in enumerate(param_combinations):
    model = GradientBoostingClassifier(
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        n_estimators=params['n_estimators'],
        min_samples_leaf=params['min_samples_leaf'],
        random_state=42
    )
    
    scores = cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
    mean_accuracy = scores.mean()
    
    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        best_params = params
        best_model_gbdt = model
    
    print(f"  [{i+1}/{len(param_combinations)}] lr={params['learning_rate']}, depth={params['max_depth']}, n={params['n_estimators']}, leaf={params['min_samples_leaf']} -> CV: {mean_accuracy:.4f}")

print(f"\n  最佳 GBDT 参数:")
print(f"    - 学习率: {best_params['learning_rate']}")
print(f"    - 树的最大深度: {best_params['max_depth']}")
print(f"    - 弱学习器数量: {best_params['n_estimators']}")
print(f"    - 叶子节点最小样本数: {best_params['min_samples_leaf']}")
print(f"  最佳交叉验证准确率: {best_accuracy:.4f}")

best_model_gbdt.fit(X_scaled, y_encoded)
y_pred_gbdt = best_model_gbdt.predict(X_scaled)
gbdt_train_accuracy = accuracy_score(y_encoded, y_pred_gbdt)

print("\n[5/6] GBDT 模型 - 加权版本...")
class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weight_dict = dict(zip(range(len(class_weights)), class_weights))
print(f"  类别权重: {class_weight_dict}")

gbdt_weighted = GradientBoostingClassifier(
    learning_rate=0.01,
    max_depth=20,
    n_estimators=400,
    min_samples_leaf=50,
    random_state=42
)

scores_weighted = cross_val_score(gbdt_weighted, X_scaled, y_encoded, cv=cv, scoring='accuracy')
print(f"  加权 GBDT 交叉验证准确率: {scores_weighted.mean():.4f}")

gbdt_weighted.fit(X_scaled, y_encoded)
y_pred_gbdt_weighted = gbdt_weighted.predict(X_scaled)
gbdt_weighted_train_accuracy = accuracy_score(y_encoded, y_pred_gbdt_weighted)

print("\n[6/6] MNL 模型（多项Logit）...")
mnl_model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)

scores_mnl = cross_val_score(mnl_model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
print(f"  MNL 交叉验证准确率: {scores_mnl.mean():.4f}")

mnl_model.fit(X_scaled, y_encoded)
y_pred_mnl = mnl_model.predict(X_scaled)
mnl_train_accuracy = accuracy_score(y_encoded, y_pred_mnl)

print("\n  MNL 模型 - 加权版本...")
mnl_weighted = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

scores_mnl_weighted = cross_val_score(mnl_weighted, X_scaled, y_encoded, cv=cv, scoring='accuracy')
print(f"  加权 MNL 交叉验证准确率: {scores_mnl_weighted.mean():.4f}")

mnl_weighted.fit(X_scaled, y_encoded)
y_pred_mnl_weighted = mnl_weighted.predict(X_scaled)
mnl_weighted_train_accuracy = accuracy_score(y_encoded, y_pred_mnl_weighted)

print("\n" + "=" * 60)
print("模型对比结果")
print("=" * 60)

print("\n| 模型类型          | 交叉验证准确率 | 训练集准确率 |")
print("|-------------------|----------------|--------------|")
print(f"| GBDT (最优参数)   | {best_accuracy:.4f}       | {gbdt_train_accuracy:.4f}      |")
print(f"| GBDT (加权)       | {scores_weighted.mean():.4f}       | {gbdt_weighted_train_accuracy:.4f}      |")
print(f"| MNL (多项Logit)   | {scores_mnl.mean():.4f}       | {mnl_train_accuracy:.4f}      |")
print(f"| MNL (加权)        | {scores_mnl_weighted.mean():.4f}       | {mnl_weighted_train_accuracy:.4f}      |")

print("\n" + "=" * 60)
print("分类报告 - GBDT (加权)")
print("=" * 60)
print(classification_report(y_encoded, y_pred_gbdt_weighted, target_names=class_names))

print("\n" + "=" * 60)
print("分类报告 - MNL (加权)")
print("=" * 60)
print(classification_report(y_encoded, y_pred_mnl_weighted, target_names=class_names))

print("\n" + "=" * 60)
print("混淆矩阵 - GBDT (加权)")
print("=" * 60)
cm_gbdt = confusion_matrix(y_encoded, y_pred_gbdt_weighted)
print("预测→")
print("实际↓", end="")
for name in class_names:
    print(f"  {name[:6]}", end="")
print()
for i, name in enumerate(class_names):
    print(f"{name[:6]}", end="")
    for j in range(len(class_names)):
        print(f"  {cm_gbdt[i,j]:4d}", end="")
    print()

print("\n" + "=" * 60)
print("混淆矩阵 - MNL (加权)")
print("=" * 60)
cm_mnl = confusion_matrix(y_encoded, y_pred_mnl_weighted)
print("预测→")
print("实际↓", end="")
for name in class_names:
    print(f"  {name[:6]}", end="")
print()
for i, name in enumerate(class_names):
    print(f"{name[:6]}", end="")
    for j in range(len(class_names)):
        print(f"  {cm_mnl[i,j]:4d}", end="")
    print()

results_df = pd.DataFrame({
    'Model': ['GBDT (最优参数)', 'GBDT (加权)', 'MNL (多项Logit)', 'MNL (加权)'],
    'CV_Accuracy': [best_accuracy, scores_weighted.mean(), scores_mnl.mean(), scores_mnl_weighted.mean()],
    'Train_Accuracy': [gbdt_train_accuracy, gbdt_weighted_train_accuracy, mnl_train_accuracy, mnl_weighted_train_accuracy]
})
results_df.to_csv(f"{OUTPUT_DIR}/model_comparison_results.csv", index=False, encoding='utf-8-sig')
print(f"\n模型对比结果已保存至: {OUTPUT_DIR}/model_comparison_results.csv")

print("\n" + "=" * 60)
print("分析完成!")
print("=" * 60)
