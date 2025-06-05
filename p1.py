import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df = data.frame

print("Dataset Shape:", df.shape)
print("Dataset Columns:", df.columns)

print("Generating Histograms")
df.hist(bins=20, figsize=(14, 10), color='skyblue', edgecolor='black')
plt.suptitle('Histograms of Numerical Features', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("\nGenerating box plots...")
plt.figure(figsize=(14, 10))

for i, col in enumerate(df.columns):
    plt.subplot(3, 3, i + 1)  # 3 rows x 3 columns grid
    plt.boxplot(df[col].dropna())
    plt.title(col)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("\nOutlier Analysis:")
for col in df.columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outliers")