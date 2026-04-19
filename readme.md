# PYTHON BASICS by Aayudh

## Table of contents    
* [Page 1: 2025-13-11](#id-section1). Chapter 1: Basic function 
* [Page 2: 2025-13-11](#id-section2). Chapter 2: Pandas basics
* [Page 3: 2025-13-11](#id-section3). Chapter 3: Biopython
* [Page 4: 2025-13-11](#id-section4). Chapter 4: Making a regressions
* [Page 5: 2026-19-04](#id-section5). Chapter 5: Machine learning

------
<div id='id-section1'/>
    
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

```
def gc(seq):
  gc_cont = seq.count("G") + seq.count("C")
  total_lenth = len(seq)
  return (gc_cont/total_lenth)*100

seq="ATCGATCGATCG"
print(gc(seq))

```

***Nested function***
```
def outer():
    def inner():
        print("This is inner function.")
    inner()
    print("This is outer function.")

outer()


def outer(a, b):
    result = a + b
    
    def inner(a, b):
        result = a - b
        print("Inner result:", result)
    
    # Call inner function
    inner(a, b)
    
    print("Outer result:", result)

# Call outer with values
outer(10, 5)

```

------
<div id='id-section2'/>
    
## Chapter 2: Pandas basics
```
import pandas as pd

# Create a simple sales dataset
data = {
    "Product": ["A", "B", "C", "A", "B", "C"],
    "Region": ["North", "South", "East", "West", "North", "South"],
    "Sales": [200, 150, 300, 250, 180, 220],
    "Month": ["Jan", "Jan", "Jan", "Feb", "Feb", "Feb"]
}

df = pd.DataFrame(data)
print(df)
```
#### Inspect the Data

#Get summary info
```
print(df.info())
```
# Show statistics
```
print(df.describe())
```
##### Make a mock clinical data

```
import pandas as pd
from io import StringIO

csv_data = """patient_id,age,sex,disease_status,biomarker_level,treatment_group,survival_time,event
1,65,M,1,8.2,A,320,1
2,54,F,0,3.1,B,450,0
3,70,M,1,9.5,A,210,1
4,45,F,0,2.8,B,500,0
5,60,M,1,7.9,A,300,1
6,50,F,0,3.5,A,480,0
7,72,M,1,10.2,B,180,1
8,38,F,0,2.1,B,520,0
9,67,M,1,8.8,A,250,1
10,49,F,0,3.0,A,470,0
11,62,M,1,7.5,B,310,1
12,55,F,0,,B,490,0
13,68,M,1,9.1,A,220,1
14,43,F,0,2.4,A,510,0
15,59,M,1,8.0,B,330,1
16,52,F,0,3.3,A,460,0
17,75,M,1,10.5,B,150,1
18,40,F,0,2.0,A,530,0
19,66,M,1,8.6,A,260,1
20,48,F,0,,B,480,0
21,63,M,1,7.8,A,310,1
22,57,F,0,3.6,B,450,0
23,69,M,1,9.7,A,200,1
24,44,F,0,2.7,A,500,0
25,61,M,1,8.1,B,320,1
26,53,F,0,3.2,A,470,0
27,71,M,1,10.0,B,170,1
28,39,F,0,2.2,B,520,0
29,64,M,1,8.4,A,280,1
30,47,F,0,3.1,A,490,0
31,58,M,1,7.6,B,340,1
32,51,F,0,3.4,A,460,0
33,73,M,1,10.3,B,160,1
34,42,F,0,2.5,B,510,0
35,60,M,1,8.3,A,300,1
36,54,F,0,3.0,A,480,0
37,68,M,1,9.2,B,230,1
38,46,F,0,2.9,B,500,0
39,65,M,1,8.7,A,270,1
40,49,F,0,3.3,A,470,0
"""

df = pd.read_csv(StringIO(csv_data))
df.head()
```
## Count NA
```
df.isna().sum()
```

### Correlation
```
df.corr(Numeric_only=TRUE)
```

**.info() → shows columns, data types, and non-null counts.**

**.describe() → gives quick stats for numeric columns (mean, std, min, max, etc.).**

#### T test

This analysis compares biomarker levels between disease **cases** and **controls** using both parametric and non-parametric statistical tests.

```
from scipy.stats import ttest_ind

t_stat, p_value = ttest_ind(df["column1"], df["column2"], nan_policy="omit")
print(p_value)
```
#### Detailed version 
```
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

cases = df.loc[df["disease_status"] == 1, "biomarker_level"]
controls = df.loc[df["disease_status"] == 0, "biomarker_level"]

# T-test
t_stat, t_p = ttest_ind(cases, controls, equal_var=False, nan_policy="omit")
print("T-test p-value:", t_p)

# Clean data
cases_clean = cases.dropna()
controls_clean = controls.dropna()

# Mann-Whitney
u_stat, u_p = mannwhitneyu(cases_clean, controls_clean, alternative="two-sided")
print("Mann-Whitney p-value:", u_p)

# Medians
print("Case median:", np.median(cases_clean))
print("Control median:", np.median(controls_clean))
```
---

## 📊 Data Subsetting

We first split the dataset into two groups:

```python
cases = df.loc[df["disease_status"] == 1, "biomarker_level"]
controls = df.loc[df["disease_status"] == 0, "biomarker_level"]
```
#### Subseting 

```
file_path = r"C:\Users\Aayudh\Downloads\Results_DVL-083025_20250905-00001.csv"
df = pd.read_csv(file_path)

filtered = df[df["Area"] > 200]
print(filtered.head())
```
#### Merging

```
import pandas as pd

df1 = pd.DataFrame({
    "ID": [1, 2, 3, 4],
    "Name": ["Aayudh", "John", "Maria", "Sara"]
})

df2 = pd.DataFrame({
    "ID": [1, 2, 3, 5],
    "Score": [90, 85, 88, 92]
})

#Inner Join (keep only matching IDs)
merged = pd.merge(df1, df2, on="ID", how="inner")
print(merged)

#Left Join (keep all rows from df1)
merged = pd.merge(df1, df2, on="ID", how="left")
print(merged)

## how= outer (keep all rows from both tables) or right (keep all rows from df2)

#If the merge column names are different
merged = pd.merge(df1, df2, left_on="UserID", right_on="ID", how="inner")
```


#### Add a row and add the means

```
# Calculate mean for each numeric column
mean_row = df.mean(numeric_only=True)

# Add a label to the row index
mean_row.name = "AVERAGE"

# Append the row to the dataframe
df_with_mean = pd.concat([df, mean_row.to_frame().T], ignore_index=False)

print(df_with_mean)
```

------
<div id='id-section3'/>

## Chapter 3: Biopython

#### Simple sequencing stuff

```
from Bio.Seq import Seq
my_seq = Seq("ATAGATC")
my_seq_compliment = my_seq.complement()
my_seq.reverse_complement()
print(my_seq_compliment)
print(my_seq[0]) 
```

```
from Bio.Seq import Seq
my_seq = Seq("GATCG")
for index, letter in enumerate(my_seq):
    print("%i %s" % (index, letter))
```

#### Load a fasta file and Display FASTQ Information
```
from Bio import SeqIO

# Path to your FASTA file
file_path = r"C:\Users\Aayudh\Downloads\ls_orchid.fasta.txt"

# Read and display information
for record in SeqIO.parse(file_path, "fasta"):
    print(f"ID: {record.id}")
    print(f"Description: {record.description}")
    print(f"Sequence length: {len(record.seq)}")
    print(f"Sequence (first 100 bases): {record.seq[:100]}")
    print("-" * 50)
```
#### Filter Reads by Average Quality

```
def filter_by_quality(input_file, output_file, min_quality=35):
    """Filter FASTQ reads by quality."""
    from Bio import SeqIO
    good_reads = []
    for record in SeqIO.parse(input_file, "fastq"):
        avg_qual = sum(record.letter_annotations["phred_quality"]) / len(record)
        if avg_qual >= min_quality:
            good_reads.append(record)
    SeqIO.write(good_reads, output_file, "fastq")
    print(f"✅ Filtered {len(good_reads)} reads saved to {output_file}")

filter_by_quality("sample.fastq", "filtered.fastq")

```

**Count Reads**
```
def count_reads(fastq_file):
    count = sum(1 for _ in SeqIO.parse(fastq_file, "fastq"))
    print(f"📊 Total reads in {fastq_file}: {count}")
    return count

count_reads("filtered.fastq")
```
```
# Simulate mapping stats
def mapping_stats_mock():
    total = 1000
    mapped = 950
    print(f"📈 Total reads: {total}, Mapped reads: {mapped}, Mapping rate: {mapped/total*100:.2f}%")

mapping_stats_mock()

# Simulate variant count
def count_variants_mock():
    variants = 42
    print(f"🧬 Total variants found: {variants}")

count_variants_mock()

```

<div id='id-section4'/>
-----
    
## Chapter 4: Making a regressions 

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
<div id='id-section5'/>
-----
    
## Chapter 5: Machine learning (ML)

# 🌲 Random Forest (Simple Example)

## 📌 What is Random Forest?

Random Forest is a machine learning model that:
- Builds many decision trees  
- Each tree makes a prediction  
- Final answer = majority vote  

Used mainly for classification problems.

---

## 📊 Example Dataset: Iris

The dataset contains:
- Sepal length  
- Sepal width  
- Petal length  
- Petal width  

Goal: predict flower type:
- Setosa  
- Versicolor  
- Virginica  

---

## 🧠 Full Code

```python
# Step 1: import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 2: load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 3: split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: create model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 5: train model
model.fit(X_train, y_train)

# Step 6: predict
y_pred = model.predict(X_test)

# Step 7: evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 8: predict one sample
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)

print("Predicted class:", prediction)
print("Predicted flower:", iris.target_names[prediction[0]])
```

---

## ⚡ Minimal Version

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print(model.predict(X_test))
```

---

## ✅ Key Lines

```python
model.fit(X_train, y_train)
model.predict(X_test)
```
