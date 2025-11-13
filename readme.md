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

----
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
```
#Get summary info
print(df.info())

# Show statistics
print(df.describe())
```
**.info() → shows columns, data types, and non-null counts.**

**.describe() → gives quick stats for numeric columns (mean, std, min, max, etc.).**

#### Subseting 

```
file_path = r"C:\Users\Aayudh\Downloads\Results_DVL-083025_20250905-00001.csv"
df = pd.read_csv(file_path)

filtered = df[df["Area"] > 200]
print(filtered.head())
```

----
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
