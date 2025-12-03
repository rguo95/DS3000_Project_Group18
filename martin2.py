# DS3000 Project - Group 18
# Predicting Chess Outcomes from Elo and Opening Selection

import pandas as pd
import numpy as np
import time  # for timing

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
)

import matplotlib.pyplot as plt


# =====================================================
# 1. LOAD DATA
# =====================================================

csv_path = r"C:\Users\Blazi\Year 3 Uni\DS3000\SemesterProject\DS3000_Project_Group18\chess_games.csv"

print("Loading CSV from:", csv_path)
df = pd.read_csv(csv_path)

print("\nRaw shape:", df.shape)
print("Columns:", df.columns.tolist())


# =====================================================
# 2. CLEANING & FEATURE ENGINEERING
# =====================================================

df = df.rename(columns={
    'WhiteElo': 'white_elo',
    'BlackElo': 'black_elo',
    'Result': 'result',
    'ECO': 'eco',
    'Opening': 'opening_name',
    'TimeControl': 'time_control'
})

df = df[['white_elo', 'black_elo', 'result', 'eco', 'opening_name', 'time_control']]
df = df.dropna(subset=['white_elo', 'black_elo', 'result'])

df['white_elo'] = pd.to_numeric(df['white_elo'], errors='coerce')
df['black_elo'] = pd.to_numeric(df['black_elo'], errors='coerce')
df = df.dropna(subset=['white_elo', 'black_elo'])

df = df[df['result'].isin(['1-0', '0-1'])].copy()

df['white_win'] = (df['result'] == '1-0').astype(int)

df['elo_diff'] = df['white_elo'] - df['black_elo']
df['avg_elo'] = (df['white_elo'] + df['black_elo']) / 2

df['white_higher'] = (df['elo_diff'] > 0).astype(int)
df['elo_diff_abs'] = df['elo_diff'].abs()
df['elo_diff_sq'] = df['elo_diff'] ** 2

df['eco'] = df['eco'].fillna('UNKNOWN').astype(str)
df['eco_code3'] = df['eco'].str[:3]
df['time_control'] = df['time_control'].fillna('UNKNOWN').astype(str)


def parse_time_control(tc: str) -> str:
    if tc in ['UNKNOWN', '?', None] or tc == '':
        return 'unknown'
    if '+' not in tc:
        return 'unknown'
    try:
        base, inc = tc.split('+')
        base = int(base)
    except ValueError:
        return 'unknown'

    if base <= 180:
        return 'bullet'
    elif base <= 600:
        return 'blitz'
    elif base <= 1800:
        return 'rapid'
    else:
        return 'classical'


df['tc_bucket'] = df['time_control'].apply(parse_time_control)

eco_counts = df['eco_code3'].value_counts()
rare_ecos = eco_counts[eco_counts < 100].index
df['eco_code3_clean'] = df['eco_code3'].where(~df['eco_code3'].isin(rare_ecos), 'OTHER')

print("\nAfter cleaning:", df.shape)
print(df.head())


# =====================================================
# 3. DOWNSAMPLE FOR ML
# =====================================================

N = 5_000_000
if len(df) > N:
    df_small = df.sample(n=N, random_state=42).copy()
    print(f"\nDownsampled to {N}")
else:
    df_small = df.copy()

print("ML dataframe shape:", df_small.shape)


# =====================================================
# 4. BUILD ML DATASET
# =====================================================

num_features = ['elo_diff', 'avg_elo', 'elo_diff_abs', 'elo_diff_sq']
cat_features = ['eco_code3_clean', 'tc_bucket', 'white_higher']

df_model = df_small[num_features + cat_features + ['white_win']].dropna()
X = df_model[num_features + cat_features]
y = df_model['white_win'].values

print("\nClass balance:", np.bincount(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# =====================================================
# 5. LOGISTIC REGRESSION MODEL
# =====================================================

preprocess_lr = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
        ('num', StandardScaler(), num_features),
    ]
)

log_reg_clf = Pipeline(steps=[
    ('preprocess', preprocess_lr),
    ('model', LogisticRegression(max_iter=2000, n_jobs=-1))
])

print("\nTraining Logistic Regression...")
start_lr = time.time()
log_reg_clf.fit(X_train, y_train)
end_lr = time.time()

training_time = end_lr - start_lr

y_pred_lr = log_reg_clf.predict(X_test)
y_prob_lr = log_reg_clf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred_lr)
auc = roc_auc_score(y_test, y_prob_lr)

print("\n=== Logistic Regression Results ===")
print(f"Training time: {training_time:.2f} sec")
print(f"Accuracy:      {acc:.4f}")
print(f"ROC AUC:       {auc:.4f}")
print(classification_report(y_test, y_pred_lr, digits=4))


# =====================================================
# 6. OPENING PERFORMANCE BY ELO BRACKET
# =====================================================

def elo_bracket(r):
    if r <= 999:
        return 'Under 999'
    elif r <= 1399:
        return '1000–1399'
    elif r <= 1799:
        return '1400–1799'
    elif r <= 2199:
        return '1800–2199'
    else:
        return '2200+'


df['elo_bracket'] = df['avg_elo'].apply(elo_bracket)

MIN_GAMES = 100
stats = (
    df.groupby(['elo_bracket', 'eco_code3'])
    .agg(
        games=('white_win', 'size'),
        white_win_rate=('white_win', 'mean')
    )
    .reset_index()
)

stats = stats[stats['games'] >= MIN_GAMES]

print("\n=== Top Openings Per Elo Bracket (by White Win Rate) ===")
for bracket in stats['elo_bracket'].unique():
    print("\nELO BRACKET:", bracket)
    print(
        stats[stats['elo_bracket'] == bracket]
        .sort_values('white_win_rate', ascending=False)
        .head(5)
    )

# NEW: most played openings per bracket
print("\n=== Most Played Openings Per Elo Bracket (by Games) ===")
for bracket in stats['elo_bracket'].unique():
    print("\nELO BRACKET:", bracket)
    print(
        stats[stats['elo_bracket'] == bracket]
        .sort_values('games', ascending=False)
        .head(5)[['elo_bracket', 'eco_code3', 'games', 'white_win_rate']]
    )


# =====================================================
# 7. ROC CURVE
# =====================================================

fpr, tpr, _ = roc_curve(y_test, y_prob_lr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--')

plt.title("ROC Curve — Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
