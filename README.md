#  Feature Engineering & Data Preprocessing

### Topic: Cleaning, Encoding, Scaling, Feature Selection

---

##  Summary

* How to handle **missing data**
* Techniques for **encoding categorical variables**
* When to use **normalization vs standardization**
* **Feature selection** methods to reduce dimensionality and improve model performance

---

## 1. What is Feature Engineering?

**Feature Engineering** is the process of **transforming raw data** into meaningful features that help a machine learning model learn better patterns.

---

###  Analogy: Cooking a Meal

> Raw ingredients (raw data) aren’t enough — you need to **chop**, **boil**, and **season** them. Feature engineering is the process of preparing and transforming your ingredients (data) into a delicious dish (well-performing model).

---

## 2. Handling Missing Data

Real-world datasets often contain **NaN (Not a Number)** or missing values. These can **break models** or introduce bias.

---

###  Techniques for Handling Missing Data

| Method                          | When to Use                                           |
| ------------------------------- | ----------------------------------------------------- |
| **Remove rows/columns**         | When the missing data is minimal                      |
| **Mean/Median/Mode Imputation** | For numerical features                                |
| **Forward/Backward Fill**       | For time-series data                                  |
| **Model-Based Imputation**      | Predict missing values using another model (advanced) |

---

###  Code Example

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'Age': [25, np.nan, 45, 35, np.nan],
    'Gender': ['Male', 'Female', np.nan, 'Female', 'Male']
})

# Drop missing values
df_drop = df.dropna()

# Fill with mean
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Fill with mode
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
```

---

###  Analogy: Filling a Survey

> You receive a survey where some answers are missing.
>
> * Drop the survey (drop row)?
> * Fill it with the average response (mean)?
> * Copy from similar responses (model-based)?
>   Different strategies work for different types of questions!

---

## 3. Encoding Categorical Variables

Models only understand **numbers**. Categorical values (like "Red", "Blue", "Green") must be converted to numerical format.

---

###  Encoding Methods

| Method               | Description                               | Use When                                        |
| -------------------- | ----------------------------------------- | ----------------------------------------------- |
| **Label Encoding**   | Assigns each category a number            | Ordinal data (e.g., Small=0, Medium=1, Large=2) |
| **One-Hot Encoding** | Creates a binary column for each category | Nominal data (e.g., color, city)                |

---

###  Code Example

```python
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.DataFrame({'Size': ['Small', 'Medium', 'Large', 'Small']})

# Label Encoding
le = LabelEncoder()
df['Size_Label'] = le.fit_transform(df['Size'])

# One-Hot Encoding
df_onehot = pd.get_dummies(df['Size'], prefix='Size')
```

---

### 🔍 Analogy: Making IDs

> Label Encoding is like giving each T-shirt size an **ID number**.
> One-Hot Encoding is like using a **checkbox system**:
> Small = \[1, 0, 0],  Medium = \[0, 1, 0],  Large = \[0, 0, 1]

---

## 4. Feature Scaling

Many models (especially distance-based ones like KNN, SVM, K-Means) are sensitive to **the scale of features**.

---

###  Scaling Techniques

| Method                              | Description                              | Use Case                                        |
| ----------------------------------- | ---------------------------------------- | ----------------------------------------------- |
| **Normalization (Min-Max Scaling)** | Rescales data to a \[0,1] range          | When data doesn’t follow normal distribution    |
| **Standardization (Z-score)**       | Centers data around 0 with unit variance | When data is approximately normally distributed |

---

###  Code Example

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

data = pd.DataFrame({'Income': [20000, 30000, 40000, 50000, 60000]})

# Normalization
minmax = MinMaxScaler()
data['Income_norm'] = minmax.fit_transform(data[['Income']])

# Standardization
std = StandardScaler()
data['Income_std'] = std.fit_transform(data[['Income']])
```

---

###  Analogy: Comparing Weights & Heights

> One person weighs 70kg and is 180cm tall. If you feed both into a model **unscaled**, the model might think **height is less important** because it has smaller numbers.
> Scaling puts them **on the same footing** — like converting everything into the same currency.

---

## 5. Feature Selection Techniques

Not all features are useful — some may be **irrelevant**, **redundant**, or even **harmful**.

---

###  Common Feature Selection Techniques

| Method                                  | Description                                         | Best For                             |
| --------------------------------------- | --------------------------------------------------- | ------------------------------------ |
| **Correlation Matrix**                  | Remove highly correlated features                   | Numerical features                   |
| **Chi-Square Test**                     | Measures dependence between categorical variables   | Classification problems              |
| **Recursive Feature Elimination (RFE)** | Trains a model and removes least important features | Works with any model                 |
| **L1 Regularization (Lasso)**           | Penalizes less important features to zero           | High-dimensional regression problems |

---

###  Code: Correlation Matrix

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
df = pd.DataFrame({
    'X1': [1, 2, 3, 4],
    'X2': [1, 2, 3, 4],
    'X3': [4, 3, 2, 1]
})

# Correlation matrix
corr = df.corr()
sns.heatmap(corr, annot=True)
plt.title("Correlation Matrix")
plt.show()
```

---

###  Code: Recursive Feature Elimination

```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=1000)
selector = RFE(model, n_features_to_select=2)
selector.fit(X, y)

print("Selected Features:", selector.support_)
```

---

###  Analogy: Packing for a Hiking Trip

> You have 20 items but your backpack can only carry 5.
> You must pick the most **useful and lightweight** items.
> Feature selection is like that: picking only the most **impactful** features for your model.

---

## 6. Summary Table

| Step                       | Purpose                       | Example Tool                       |
| -------------------------- | ----------------------------- | ---------------------------------- |
| **Missing Value Handling** | Fix incomplete data           | `.dropna()`, `.fillna()`           |
| **Encoding**               | Convert categories to numbers | `LabelEncoder`, `pd.get_dummies()` |
| **Scaling**                | Normalize feature ranges      | `MinMaxScaler`, `StandardScaler`   |
| **Selection**              | Keep best features            | `RFE`, `corr()`, `chi2`            |

---

## 7. Final Analogy Recap

| Analogy              | Concept                    |
| -------------------- | -------------------------- |
| Cooking              | Raw data → usable features |
| Survey with blanks   | Missing data               |
| Checkbox form        | One-Hot Encoding           |
| Different currencies | Feature scaling            |
| Packing a bag        | Feature selection          |

---
