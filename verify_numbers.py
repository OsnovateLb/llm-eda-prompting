"""Пересчёт чисел из report.md: python verify_numbers.py (нужны pandas и scipy)."""

import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("data/breast_cancer_wisconsin.csv")
num = df.select_dtypes("number")
feats = ["radius", "texture", "perimeter", "area", "smoothness", "compactness",
         "concavity", "concave_points", "symmetry", "fractal_dimension"]

print("== Промпт 1: структура")
print("размер:", df.shape, "| пропуски:", int(df.isna().sum().sum()), "| дубликаты:", int(df.duplicated().sum()))
print("классы:", df.diagnosis.value_counts().to_dict())
zero_cols = [c for c in num if "concav" in c]
zero_rows = (num[zero_cols] == 0).all(axis=1)
print("строк с нулями во всех concav*-столбцах:", int(zero_rows.sum()), df.loc[zero_rows, "diagnosis"].value_counts().to_dict())
print("нарушений worst >= mean:", sum(int((df[f"worst_{f}"] < df[f"mean_{f}"]).sum()) for f in feats))
ratio = df.mean_area / (np.pi * df.mean_radius ** 2)
print(f"area/(pi*r^2): среднее {ratio.mean():.3f}, мин {ratio.min():.3f}, макс {ratio.max():.3f}")

print("\n== Промпт 2: различия классов")
rows = []
for c in num:
    m, b = df.loc[df.diagnosis == "M", c], df.loc[df.diagnosis == "B", c]
    d = (m.mean() - b.mean()) / np.sqrt((m.var() + b.var()) / 2)
    auc = stats.mannwhitneyu(m, b).statistic / (len(m) * len(b))
    rows.append((c, b.mean(), m.mean(), m.mean() / b.mean(), d, auc))
res = pd.DataFrame(rows, columns=["feature", "B", "M", "M/B", "cohen_d", "AUC"]).sort_values("AUC", ascending=False)
print(res.round(3).to_string(index=False))
corr = num.corr().where(np.triu(np.ones((30, 30)), 1).astype(bool)).stack()
print("пар с |r| > 0.9:", int((corr.abs() > 0.9).sum()))
print(corr.sort_values(ascending=False).head(3).round(3))

print("\n== Промпт 3: аномалии")
q1, q3 = num.quantile(0.25), num.quantile(0.75)
iqr = q3 - q1
out15 = (num < q1 - 1.5 * iqr) | (num > q3 + 1.5 * iqr)
out3 = (num < q1 - 3 * iqr) | (num > q3 + 3 * iqr)
print("строк с выбросом 1.5*IQR:", int(out15.any(axis=1).sum()), "| ячеек:", int(out15.sum().sum()), "| 3*IQR строк:", int(out3.any(axis=1).sum()))
print("по классам:", df.loc[out15.any(axis=1), "diagnosis"].value_counts().to_dict())
mad = (num - num.median()).abs().median() * 1.4826
rz = ((num - num.median()) / mad.replace(0, np.nan)).abs()
for i in rz.max(axis=1).sort_values(ascending=False).head(5).index:
    col = rz.loc[i].idxmax()
    print(f"строка {i} ({df.loc[i, 'diagnosis']}): {col} = {df.loc[i, col]}, робастный z = {rz.loc[i, col]:.1f}")
same = [i for i in df.index if all(df.loc[i, f"worst_{f}"] == df.loc[i, f"mean_{f}"] for f in feats)]
print("строки, где все worst == mean:", same)

print("\n== Промпт 4: пороговые правила")
for col in ["worst_perimeter", "worst_radius", "worst_concave_points"]:
    best_t = max(np.unique(df[col]), key=lambda t: (np.where(df[col] > t, "M", "B") == df.diagnosis).mean())
    pred = np.where(df[col] > best_t, "M", "B")
    is_m = df.diagnosis == "M"
    print(f"{col} > {best_t}: точность {(pred == df.diagnosis).mean():.3f}, "
          f"чувствительность {((pred == 'M') & is_m).sum() / is_m.sum():.3f}, "
          f"специфичность {((pred == 'B') & ~is_m).sum() / (~is_m).sum():.3f}, пропущено M: {int(((pred == 'B') & is_m).sum())}")
