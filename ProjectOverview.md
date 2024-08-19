![image](https://github.com/DeandraShae/Airbnb-Analysis-Project/assets/144077177/adf861b6-52ea-4aa1-8b6d-16ef6ecb37b8)
# **Project Overview**

**Objective:** Segment Airbnb customers based on their booking habits and preferences.

**Goal:** Develop targeted marketing strategies for each segment to enhance booking rates and customer satisfaction.

**Repository:** [Github — Airbnb Analysis Project](http://github.com/DeandraShae/Airbnb-Analysis-Project)

# **Reasoning for Project Choice:**

This was my first portfolio project, chosen to delve into real-world data and apply data analysis techniques to extract actionable insights. As I am leaning towards product analyst tasks, the Airbnb dataset provides a rich source of information that allows for a comprehensive analysis of customer behavior and preferences. By working on this project, I aimed to:

- **Develop Practical Skills:** Enhance my skills in data cleaning, preprocessing, and exploratory data analysis (EDA).
- **Apply Advanced Techniques:** Use statistical and machine learning techniques to derive meaningful conclusions.

# **Takeaways from the Project:**

I would have liked to delve deeper into constructing more models and performing more feature engineering. I will focus on these aspects in future projects and use Tableau for visualizations.

# **Packages and Dependencies**

```python
# Data manipulation and analysis
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning and statistical modeling
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.formula.api import ols
from xgboost import XGBRegressor
