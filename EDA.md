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
![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/9bd70a48-8bed-4609-bbeb-250f31d9d2aa/8d583a5e-3b1f-4474-8200-4eec46ddba6f/Untitled.png)

**Interpretation:** The plot highlights that higher-priced listings are predominantly in central areas, while lower-priced listings are more spread out. This information can help in understanding the spatial and economic landscape of Airbnb listings in New York City.

## **Distribution of Last Review Dates:**

**Time Series Plot:** Highlighted activity peaks, with a notable drop in 2020 due to the COVID-19 pandemic.

![Time Series Plot](https://miro.medium.com/v2/resize:fit:560/0*4gWHAWJdrsMeMHtr)
