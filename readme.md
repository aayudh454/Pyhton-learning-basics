## Chapter 1: Basic function
**Write a loop**
```
for i in range(10):
    print("iterations:", i)
```
**While argument**
```
count =5
while count <5:
    print("count", count)
    count +=1 
```

**If-else statement**
```
num = 14

if num>10:
    print ("more than 10")
elif num ==0:
    print("zero")
else:
    print("negative")
```
#### check if a number is odd or even
```
import random
num = random.randint(1, 100)


# Check if the number is even or odd
if num % 2 == 0:
    print(f"{num} is even.")
else:
    print(f"{num} is odd.")
```


**Function**
```
def function_name(parameters):
    # code block
    return value
```
```
def add(a, b):
    result = a + b
    return result

print(add(5, 7))
```

Nested function
```
def outer():
    def inner():
        print("This is inner function.")
    inner()
    print("This is outer function.")

outer()

```

-----
## Chapter 2: Making a regressions 

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
