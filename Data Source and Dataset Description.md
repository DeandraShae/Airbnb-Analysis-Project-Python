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


