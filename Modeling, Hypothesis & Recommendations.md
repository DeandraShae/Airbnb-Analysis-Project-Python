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
