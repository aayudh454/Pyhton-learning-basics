**libraries you need**

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
```
**Load a file**

```
df = pd.read_csv(r"C:\Users\Aayudh\Downloads\Results_DVL-083025_20250905-00001.csv")
print(df.head())
```

**Linear regression**

```
X = df[["Area"]]
y = df["Mean"]

model = LinearRegression()
model.fit(X,y)

# Predict values
y_pred = model.predict(X)

# Calculate R²
r2 = r2_score(y, y_pred)
print(f"R² = {r2:.3f}")

plt.scatter(df["Area"], df["Mean"], color = "red", alpha =.6)
plt.plot(df["Area"], y_pred, color="black", linewidth=2, label="Regression line")
plt.xlabel("Area")
plt.ylabel("Mean")
plt.legend()
plt.text(
    0.05, 0.95, 
    f"$R^2$ = {r2:.3f}",
    transform=plt.gca().transAxes,   # position relative to axes (not data)
    fontsize=12,
    verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray')
)

plt.show()
```
