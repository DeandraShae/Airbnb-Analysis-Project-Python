![image](https://github.com/DeandraShae/Airbnb-Analysis-Project/assets/144077177/adf861b6-52ea-4aa1-8b6d-16ef6ecb37b8)
# **Project Overview**

**Objective:** Segment Airbnb customers based on their booking habits and preferences.

**Goal:** Develop targeted marketing strategies for each segment to enhance booking rates and customer satisfaction.

**Repository:** [Github — Airbnb Analysis Project](http://github.com/DeandraShae/Airbnb-Analysis-Project)

# **Reasoning for Project Choice:**

- **Develop Practical Skills:** Enhance my skills in data cleaning, preprocessing, and exploratory data analysis (EDA).
- **Apply Advanced Techniques:** Use statistical and machine learning techniques to derive meaningful conclusions.

# Table of Contents

1. [Project Overview](#project-overview)
2. [Reasoning for Project Choice](#reasoning-for-project-choice)
3. [Packages and Dependencies](#packages-and-dependencies)
4. [Data Source and Dataset Description](#data-source-and-dataset-description)
5. [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
6. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [Univariate Analysis](#univariate-analysis)
   - [Bivariate Analysis](#bivariate-analysis)
   - [Geographic Visualization](#geographic-visualization)
   - [Distribution of Last Review Dates](#distribution-of-last-review-dates)
7. [Modeling](#modeling)
   - [Hypothesis: Effect of House Rules on Customer Choice](#hypothesis-effect-of-house-rules-on-customer-choice)
   - [Hypothesis: Room Type Impact on Availability and Number of Reviews](#hypothesis-room-type-impact-on-availability-and-number-of-reviews)
8. [Results and Interpretation](#results-and-interpretation)
9. [Discussion](#discussion)
10. [Limitations](#limitations)
11. [Conclusion and Recommendations](#conclusion-and-recommendations)
12. [Future Work](#future-work)
13. [References](#references)
14. [Appendices](#appendices)
    - [Code](#code)
    - [Additional Resources](#additional-resources)


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
```
# **Data Source and Dataset Description**

**Source:** [Airbnb Open Data on Kaggle](https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata/data)

**Dataset Description:**  
Airbnb Inc. operates an online marketplace for lodging, primarily homestays for vacation rentals and tourism activities. Based in San Francisco, California, the platform is accessible via a website and mobile app. Airbnb does not own any listed properties; instead, it profits by receiving a commission from each booking. The company was founded in 2008, and its name is a shortened version of AirBedandBreakfast.com.

# **Data Cleaning and Preprocessing**

1. **Initial Inspection:**  
   Used `.head()`, `.info()`, and `.describe()` for data structure, types, and potential issues.

2. **Handling Missing Values:**
   - Replaced NaN values in categorical data like `host_identity_verified` with logical values.
   - Columns like `last review`, `reviews per month`, and `license` show missing values.
   - Filled missing values for `reviews per month` with 0, assuming no reviews.
   - Dropped `license` due to irrelevance.

3. **Correcting Data Types:**

    ```python
    # Convert columns to appropriate data types
    airbnb_df['last_review'] = pd.to_datetime(airbnb_df['last_review'])
    airbnb_df['host_identity_verified'] = airbnb_df['host_identity_verified'].astype('bool') 
    ```

   - Ensured date columns are in datetime format.
   - Converted categorical columns like `host_identity_verified` to boolean type.
   - Ensured proper handling of date operations and numerical conversions.

4. **Dealing with Outliers:**

    ```python
    # Cap extreme values
    Q1 = airbnb_df['minimum_nights'].quantile(0.25)
    Q3 = airbnb_df['minimum_nights'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    airbnb_df = airbnb_df[(airbnb_df['minimum_nights'] >= lower_bound) & (airbnb_df['minimum_nights'] <= upper_bound)] 
    ```

   - Used IQR for numerical columns like `minimum_nights` and the `number of reviews` to detect outliers.
   - Capped extreme values to reduce the impact on analysis.

5. **Standardizing Text Data:**

    ```python
    # Standardize text data
    airbnb_df['neighbourhood'] = airbnb_df['neighbourhood'].str.lower().str.replace('[^a-zA-Z\s]', '') 
    ```

   - Addressed inconsistencies in text data, ensuring consistent capitalization and removing special characters or typos.
   - Standardized text variables and mapped boolean values.

6. **Verifying Unique Identifiers:**

    ```python
    # Check for unique identifiers
    airbnb_df.drop_duplicates(subset=['id', 'host_id'], inplace=True) 
    ```

   - Ensured `id` and `host_id` columns are unique, handling duplicates as needed.

7. **Normalization/Standardization (Optional):**

    ```python
    # Normalize numerical fields
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    airbnb_df[['price', 'service_fee']] = scaler.fit_transform(airbnb_df[['price', 'service_fee']]) 
    ```

   - Normalized or standardized numerical fields like `price` for predictive modeling readiness. Used the standardized option for the variable `price`.

8. **Feature Engineering:**

    ```python
    # Create new features
    airbnb_df['days_since_last_review'] = (pd.to_datetime('today') - airbnb_df['last_review']).dt.days 
    ```

   - Created new features from existing data to enhance analysis, such as deriving days since `last review`.
  
   # **Exploratory Data Analysis (EDA)**

## **Univariate Analysis:**

- **Histograms and Box Plots:** Visualized key variables like **price, minimum nights, number of reviews**, and **availability_365**.
- **Price:** Most listings are affordable, with a few expensive outliers.
- **Minimum Nights:** Majority require only a few nights, with rare extreme values.
- **Number of Reviews:** Skewed towards fewer reviews, indicating many listings are either new or infrequently reviewed.
- **Availability:** Two peaks near **0 and 365 days**, indicating diverse listing availability.

![Histogram](https://miro.medium.com/v2/resize:fit:560/0*2WIt-0T1OlIj3DJO)
![Box Plot](https://miro.medium.com/v2/resize:fit:560/0*h4Tg6M-RKsjo2MK6)

## **Bivariate Analysis:**

- **Scatter Plots and Correlation Heatmap:**
- **Price vs. Other Variables:** No strong relationships were observed, though trends were identified.
- **Minimum Nights vs. Other Variables:** Longer stays generally have fewer reviews.
- **Number of Reviews vs. Availability:** Listings with more reviews tend to have higher availability.

![Scatter Plot](https://miro.medium.com/v2/resize:fit:560/0*QMZoeXc56e1cXZFp)
![Correlation Heatmap](https://miro.medium.com/v2/resize:fit:560/0*Mk0WBb7_7wajMUt-)

# **Geographic Visualization:**

**Scatter Plot:** Visualized distribution of listings based on latitude and longitude.

**Observations:**

- **Clusters of Listings:**
  - Dense clusters are visible in Manhattan, Brooklyn, and parts of Queens.
  - Fewer listings are found in the outer areas of the city.
- **Price Variation:**
  - Higher-priced listings (yellow) are mostly concentrated in central areas like Manhattan.
  - Lower-priced listings (purple) are more evenly distributed but still show some clustering.

![Geographic Visualization](https://miro.medium.com/v2/resize:fit:560/0*6c9c3aXU6bect5X_)
![Geographic Visualization](https://miro.medium.com/v2/format:webp/0*pvzRamPUxLnaWn1X)

**Interpretation:** The plot highlights that higher-priced listings are predominantly in central areas, while lower-priced listings are more spread out. This information can help in understanding the spatial and economic landscape of Airbnb listings in New York City.

## **Distribution of Last Review Dates:**

**Time Series Plot:** Highlighted activity peaks, with a notable drop in 2020 due to the COVID-19 pandemic.

![Time Series Plot](https://miro.medium.com/v2/resize:fit:560/0*4gWHAWJdrsMeMHtr)

**Modeling**
============

**1\. Hypothesis: Effect of House Rules on Customer Choice**
------------------------------------------------------------

**Hypothesis:** Strict house rules negatively impact the likelihood of a property being booked.

**Analysis Approach:**

*   **Categorization of House Rules:** House rules were categorized into lenient, moderate, and strict based on the language or specific rules mentioned.
*   **Comparison of Booking Rates:** Booking rates for each category were calculated and compared.
*   **Correlation Analysis:** A Chi-square test was performed to determine if there’s a correlation between the strictness of house rules and booking rates.

 ```python
# ANOVA test for house rules
import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols('number_of_reviews ~ C(house_rules)', data=airbnb_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
 ```

**Results:**

*   **ANOVA Test Results:** The ANOVA results showed a statistically significant difference in the _**number of reviews**_ and _**availability\_365**_ across different _**house rule categories**_.
*   **Number of Reviews:** The F-value was 67.77 with a p-value of 8.885×10−448.885 \\times 10^{-44}8.885×10−44, indicating a significant difference in the _**number of reviews**_ based on _**house rule categories**_.
*   **Availability\_365:** The F-value was 17.15 with a p-value of 3.942×10−113.942 \\times 10^{-11}3.942×10−11, indicating a significant difference in availability based on house rule categories.

**Interpretation:**

*   **Number of Reviews:** Listings with stricter house rules tend to have fewer reviews, suggesting that strict rules might discourage bookings or reviews.
*   **Availability:** Listings with strict house rules tend to have lower availability, indicating that stricter rules might lead to fewer bookings.

**2\. Hypothesis: Room Type Impact on Availability and Number of Reviews**
--------------------------------------------------------------------------

*   **Hypothesis:** Different room types have a significant effect on the availability and the number of reviews.

**Analysis Approach:**

*   **Formulation of Hypothesis:**
*   **Null Hypothesis (H0):** No significant difference exists in the average number of reviews among different room types.
*   **Alternative Hypothesis (Ha):** There is a significant difference in the average number of reviews among different room types.
*   **ANOVA Test:** Conducted ANOVA tests to determine the impact of room type on availability and number of reviews.

```python
# ANOVA test for room type
model = ols('availability_365 ~ C(room_type)', data=airbnb_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)ppyt  
```

**Results:**

**Availability\_365:**

*   **Sum of Squares (room\_type):** 3.793756×1063.793756 \\times 10⁶³.793756×106
*   **F-value:** 71.167182
*   **p-value:** 5.693176×10−465.693176 \\times 10^{-46}5.693176×10−46

**Number of Reviews:**

*   **Sum of Squares (room\_type):** 1.138342×1051.138342 \\times 10⁵¹.138342×105
*   **F-value:** 19.745657
*   **p-value:** 8.626886×10−138.626886 \\times 10^{-13}8.626886×10−13

**Interpretation:**

*   **Effect on Availability:** The very _**low p-value**_ indicates a statistically significant effect of room type on availability. This means that the type of room (e.g., entire home, private room) influences how often the listing is available throughout the year.
*   **Effect on Number of Reviews:** Similarly, the _**very low p-value**_ indicates a significant effect of room type on the number of reviews. This suggests that the room type influences the likelihood of guests leaving reviews.

**Results and Interpretation**
==============================

**Findings:**
-------------

1.  **Effect of House Rules:**

*   Listings with stricter house rules tend to have fewer reviews, suggesting that strict rules might discourage bookings or reviews.
*   Listings with strict house rules tend to have lower availability, indicating that stricter rules might lead to fewer bookings.

**2\. Effect of Room Type:**

*   **Availability:** The very low p-value indicates a statistically significant effect of room type on availability. This means that the type of room (e.g., entire home, private room) influences how often the listing is available throughout the year.
*   **Number of Reviews:** Similarly, the very low p-value indicates a significant effect of room type on the number of reviews. This suggests that the room type influences the likelihood of guests leaving reviews.

**Discussion:**
===============

The models accurately predicted the effects of house rules and room types on Airbnb listings. Stricter house rules correlate with fewer reviews and lower availability, while different room types significantly affect availability and the number of reviews. These insights suggest that modifying house rules and optimizing room types can influence customer behavior and booking rates.

**Limitations:**
================

*   Potential biases due to missing data and the assumption of uniform behavior across different segments.

**Conclusion and Recommendations**
==================================

**Conclusions:** The analysis highlighted key factors influencing Airbnb customer behavior, such as house rules and room types. These insights can help tailor marketing and operational strategies to different segments, improving booking rates and customer satisfaction.

**Recommendations:**
====================

1.  **Revise House Rules:** Consider the impact of strict house rules on booking rates and adjust policies to balance flexibility and property protection. Stricter rules may need to be softened to attract more bookings and reviews.
2.  **Optimize Room Types:** Adjust pricing and marketing strategies based on room type to maximize occupancy and reviews. For example, entire homes might be marketed differently than private or shared rooms to better align with customer preferences and behavior.
3.  **Enhance Listings with Higher Service Fees:** Although not included in the final modeling, the initial insights suggested that listings with higher service fees tend to have better amenities and reviews. Promoting these listings could attract more customers seeking higher-quality accommodations.

**Future Work:**
================

*   Further research could explore the impact of additional features such as location-specific amenities, seasonal trends, and long-term booking patterns. Investigating these factors can provide deeper insights into customer preferences and improve the accuracy of predictive models.

**References**
==============

**Datasets:**
=============

*   Source: [**Kaggle**](https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata/data)
*   Cleaned Dataset: [**Github**](https://github.com/DeandraShae/Airbnb-Analysis-Project)

**Libraries and Tools:**
========================

*   Python (Pandas, NumPy, SciPy, Scikit-Learn)

**Academic References:**
========================

*   Relevant academic papers and articles on customer segmentation and behavior analysis.

**Appendices**
==============

**Code:**
=========

*   [Github — Airbnb Analysis Project](http://github.com/DeandraShae/Airbnb-Analysis-Project)
*   **Linear Regression:**
```python
    from sklearn.linear_model import LinearRegression  
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Prepare data
    X = airbnb_df[['service_fee', 'number_of_reviews', 'reviews_per_month']]
    y = airbnb_df['price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
 ```
    
```python
*   **Ridge Regression,** Lasso Regression, XGBoost,

    from sklearn.linear_model import Ridge 
    
    # Train model
    ridge_model = Ridge()
    ridge_model.fit(X_train, y_train)
    
    # Predict
    y_pred_ridge = ridge_model.predict(X_test)
    
    # Evaluate
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    print(f'Ridge Mean Squared Error: {mse_ridge}')
    print(f'Ridge R^2 Score: {r2_ridge}')
    
    from sklearn.linear_model import Lasso
    
    # Train model
    lasso_model = Lasso()
    lasso_model.fit(X_train, y_train)
    
    # Predict
    y_pred_lasso = lasso_model.predict(X_test)
    
    # Evaluate
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)
    r2_lasso = r2_score(y_test, y_pred_lasso)
    print(f'Lasso Mean Squared Error: {mse_lasso}')
    print(f'Lasso R^2 Score: {r2_lasso}')
    
    from xgboost import XGBRegressor
    
    # Train model
    xgb_model = XGBRegressor(learning_rate=0.1, max_depth=3, n_estimators=100)
    xgb_model.fit(X_train, y_train)
    
    # Predict
    y_pred_xgb = xgb_model.predict(X_test)
    
    # Evaluate
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    print(f'XGBoost Mean Squared Error: {mse_xgb}')
    print(f'XGBoost R^2 Score: {r2_xgb}')
```

**Additional Resources:**
=========================

*   Supplementary materials and data visualizations support the project

![Additional Visual](https://miro.medium.com/v2/resize:fit:640/format:webp/0*UVsZR9o6AeHVLbzd)
