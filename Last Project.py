#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import math


# In[2]:


df = pd.read_csv('/Users/siminjahankhah/Desktop/Master_School/Final_Mastery/TravelTide_dataset.csv')


# In[3]:


df.info()


# In[4]:


df.describe()


# # Outliers

# ## Box Plots before removing the outliers

# In[5]:


plt.figure(figsize=(8, 6))
sns.boxplot(x='hotel_per_room_usd', data=df)

# Show the plot
plt.show()


# In[6]:


plt.figure(figsize=(8, 6))
sns.boxplot(x='flight_base_fare_usd', data=df)

# Show the plot
plt.show()


# In[7]:


plt.figure(figsize=(8, 6))
sns.boxplot(x='page_clicks', data=df)

# Show the plot
plt.show()


# In[8]:


# Calculate the IQR for hotel_per_room_usd
Q1 = df['hotel_per_room_usd'].quantile(0.25)
Q3 = df['hotel_per_room_usd'].quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

upper_bound


# In[9]:


# Calculate the IQR for flight_base_fare_usd
Q1 = df['flight_base_fare_usd'].quantile(0.25)
Q3 = df['flight_base_fare_usd'].quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

upper_bound


# Use Domain Knowledge: Depending on our domain and the specific context of our data, we have insights into what constitutes a reasonable or valid data point. We can use our domain knowledge to filter out data points that are outliers in a practical sense.

# In[10]:


# Create a boolean mask for outliers
outlier_mask = (df['hotel_per_room_usd'] > 900) | (df['flight_base_fare_usd'] > 6500) | (df['page_clicks']> 250)
# Create a subset of outliers
outliers_df = df[outlier_mask]
# Create a DataFrame of non-outliers
df = df[~outlier_mask]


# # Box Plot after removing the outliers

# In[11]:


plt.figure(figsize=(8, 6))
sns.boxplot(x='hotel_per_room_usd', data=df)

# Show the plot
plt.show()


# In[12]:


plt.figure(figsize=(8, 6))
sns.boxplot(x='flight_base_fare_usd', data=df)

# Show the plot
plt.show()


# In[13]:


plt.figure(figsize=(8, 6))
sns.boxplot(x='page_clicks', data=df)

# Show the plot
plt.show()


# In[14]:


plt.figure(figsize=(8, 6))
sns.histplot(x='hotel_per_room_usd', data=df)

# Show the plot
plt.show()


# # Distibution of Discount Amount

# In[15]:


# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot the first histogram
sns.histplot(data=df, x='flight_discount_amount', bins=12, ax=axes[0])
axes[0].set_title('Flight Discount Amount')

# Plot the second histogram
sns.histplot(data=df, x='hotel_discount_amount', bins=9, ax=axes[1])
axes[1].set_title('Hotel Discount Amount')

# Find the maximum count in both histograms
max_count = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])

# Set a common y-axis limit
for ax in axes:
    ax.set_ylim(0, max_count)

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()


# # Customer Booking Behavior Analysis

# In[16]:


condition1 = (df['flight_booked'] == False) & (~df['hotel_per_room_usd'].isnull())
percentage1 =condition1.sum()/ len(df) *100
condition2 = (df['flight_booked'] == False) & (df['hotel_per_room_usd'].isnull())
percentage2 =condition2.sum()/ len(df) *100
condition3 = (df['flight_booked'] == True) & (~df['hotel_per_room_usd'].isnull())
percentage3 =condition3.sum()/ len(df) *100
condition4 = (df['flight_booked'] == True) & (df['hotel_per_room_usd'].isnull())
percentage4 =condition4.sum()/ len(df) *100

# Display the resulting DataFrame
print(f"The percentage of customers who didn't book a flight but booked a hotel is: {percentage1:.2f}%")
print(f"The percentage of customers who didn't book a flight and didn't book a hotel is: {percentage2:.2f}%")
print(f"The percentage of customers who booked a flight and hotel is: {percentage3:.2f}%")
print(f"The percentage of customers who booked a flight but didn't book a hotel is: {percentage4:.2f}%")


# # Data Aggregation

# In[17]:


# To have float numbers for 'hotel_per_room_usd'
pd.options.display.float_format = '{:.2f}'.format


# In[18]:


custom_agg = lambda x: x.nunique()
custom_agg.__name__ = 'unique_count'

#The aggregation functions

numeric_aggregations = {
    'flight_discount_amount': ['sum', 'mean', 'min', 'max',custom_agg],
    'hotel_discount_amount': ['sum', 'mean', 'min', 'max',custom_agg],
    'flight_seats':['sum', 'mean', 'min', 'max',custom_agg],
    'hotel_rooms':['sum', 'mean', 'min', 'max',custom_agg],
    'hotel_per_room_usd':['sum', 'mean', 'min', 'max',custom_agg],
    'page_clicks': ['sum', 'mean','min', 'max',custom_agg],
    'flight_base_fare_usd' : ['sum', 'mean','min', 'max',custom_agg]}
result1 = df.agg(numeric_aggregations)
result1


# In[19]:


#Number of unique values for 'user_id','trip_id', and 'home_country'
object_aggregations = {
    'user_id': 'nunique',
    'trip_id': 'nunique',
    'home_country': 'nunique'}

result2 = df.agg(object_aggregations)
result2


# In[20]:


# Sum the binary values (1 for True, 0 for False)
Boolian_aggregations = {'flight_booked': 'sum',
    'hotel_booked': 'sum',
    'cancellation': 'sum',
    'flight_discount' : 'sum',
    'hotel_discount' : 'sum'}
result3 = df.agg(Boolian_aggregations)
result3


# We need to have a column without any NaN values for K-means clustering, so we should refill NaN values with 0 for flight_discount_amount and hotel_discount_amount because NaN values mean customer hove not received any discount.

# In[21]:


df = df.copy()  # Create a copy of the DataFrame

# Fill NaN values with 0 in specific columns
df.loc[:, 'flight_discount_amount'] = df['flight_discount_amount'].fillna(0)
df.loc[:, 'hotel_discount_amount'] = df['hotel_discount_amount'].fillna(0)


# # Data Encoding

# We will work with plots that require numerical input.So we have to encode boolean columns to integers, typically 0 for False and 1 for True.

# In[22]:


df = df.copy()  # Create a copy of the DataFrame

boolean_columns = ['flight_discount', 'hotel_discount', 'flight_booked', 'hotel_booked', 'cancellation', 'married', 'has_children']

# Create a new DataFrame with the selected columns as integers
int_df = df[boolean_columns].astype(int)

# Assign the integer DataFrame back to the original DataFrame using .loc
df.loc[:, boolean_columns] = int_df.copy()


# ## Calculating Total Discount Amount and Total Purchase Metrics

# In[23]:


# Create two new columns 'total_discount_amount' and 'total_purhcase'
df['flight_discount_usd'] = df['flight_base_fare_usd']*df['flight_discount_amount']
df['hotel_discount_usd'] = df['hotel_per_room_usd']* df['hotel_discount_amount']

# Total purchase
df['total_purchase'] = df['flight_base_fare_usd'] + df['hotel_per_room_usd']
df.loc[:, 'total_purchase'] = df['total_purchase']


# ## session duration in seconds

# In[24]:


# Convert 'session_start' and 'session_end' columns to datetime objects
df['session_start'] = pd.to_datetime(df['session_start'])
df['session_end'] = pd.to_datetime(df['session_end'])

# Calculate the session duration in seconds
df['session_duration_seconds'] = (df['session_end'] - df['session_start']).dt.total_seconds()


# # EDA

# ## Correlation

# In[25]:


selected_variables = ['flight_discount_usd', 'hotel_discount_usd', 'flight_booked', 'hotel_booked', 'page_clicks',
                     'cancellation', 'flight_base_fare_usd', 'hotel_per_room_usd']
selected_corr_matrix = df[selected_variables].corr()
sns.heatmap(selected_corr_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap of Selected Variables')
plt.show()


# Both correlations between 'flight_booked', 'page_clicks'(0.41) and 'page_clicks', 'cancellation'(0.59) seem to suggest that users who interact more with TravelTide platform (measured by 'page_clicks') are more likely to both book hotels and cancel bookings. While these correlations are positive, it's important to recognize that correlation doesn't imply causation, and the relationships can be complex.

# ## PairPlot

# In[26]:


import seaborn as sns

sns.pairplot(df[['flight_discount_amount', 'hotel_discount_amount', 'flight_booked', 'hotel_booked', 'page_clicks',
                     'cancellation', 'flight_base_fare_usd', 'hotel_per_room_usd', 'married' , 'has_children']])
plt.show()


# ## scaled_ADS_per_km Metric

# This metric will give us dollars saved per kilometer traveled. There’s just one problem - distance is not natively available in the TravelTide database!
# But we do have latitude and longitude for both origin and destination points, we should define a function called haversine_distance that calculates the distance between two points on a sphere. 

# In[27]:


#Scaled Average Dollars Saved per Kilometer (Scaled ADS per km)
def haversine_distance(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Calculate the differences in latitude and longitude
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Calculate the distance
    distance = R * c

    return distance

df['distance_km'] = df.apply(lambda row: haversine_distance(row['home_airport_lat'], row['home_airport_lon'], row['flight_destination_airport_lat'], row['flight_destination_airport_lon']), axis=1)

df['debiased_ADS'] = (df['flight_discount_amount'] * df['flight_base_fare_usd']) / df['distance_km']

print('Number of missing values for debiased_ADS column : ', df['debiased_ADS'].isna().sum()) 


# We excluded the "debiased_ADS" metric from our clustering analysis. It has 35553 NaN values. Missing data can affect the performance and interpretation of clustering algorithms and we have other meaningful metrics to work with.

# ## Create new column as Age_group

# In[28]:


from datetime import datetime

# Convert 'birthdate' to datetime
df['birthdate'] = pd.to_datetime(df['birthdate'])

# Calculate age based on birthdate and current date
current_date = datetime.now()
df['age'] = (current_date - df['birthdate']).astype('<m8[Y]').astype(int)

# Define age group categories, including "Under 18"
age_bins = [0, 18, 25, 35, 45, 55, 100]
age_labels = ['Under 18', '18-25', '26-35', '36-45', '46-55', '56+']

# Bin the 'age' column into age groups
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)


# # Behavioural Metrics

# In[29]:


avrg_total_purchase = df['total_purchase'].mean()
average_page_clicks = df['page_clicks'].mean()
avg_hotel_discount_usd = df['hotel_discount_usd'].mean()
avrg_flight_discount_usd = df['flight_discount_usd'].mean()
avrg_checked_bag = df['flight_checked_bags'].mean()

print('average purchase : ', avrg_total_purchase)
print('average page clicks : ', average_page_clicks)
print('average hotel discount usd : ', avg_hotel_discount_usd)
print('average flight discount usd : ', avrg_flight_discount_usd)
print('average checked bags : ', avrg_checked_bag)


# # Data Aggregation

# In[30]:


# Select the relevant columns to aggregate for categorization and Thresholding
columns_to_aggregate = ['flight_discount_usd', 'hotel_discount_usd', 'page_clicks',
                        'flight_base_fare_usd', 'hotel_per_room_usd', 'total_purchase','hotel_booked',
                        'flight_booked', 'flight_checked_bags']

aggregated_data = df.groupby('user_id').agg({
    'age_group': lambda x: x.mode().iloc[0],
    'has_children': lambda x: x.mode().iloc[0],
    'gender': lambda x: x.mode().iloc[0],
    'flight_discount_usd': 'mean',
    'hotel_discount_usd': 'mean',
    'page_clicks': 'mean',
    'flight_base_fare_usd': 'mean',
    'hotel_per_room_usd': 'mean',
    'total_purchase': 'mean',
    'hotel_booked' : 'sum',
    'flight_booked' : 'sum',
    'flight_checked_bags' : 'mean'}).reset_index()


# In[31]:


aggregated_data


# In[32]:


# Select the relevant columns to aggregate for k-means clustering
columns_to_aggregate = ['flight_discount_usd', 'hotel_discount_usd', 'page_clicks',
                      'flight_base_fare_usd', 'hotel_per_room_usd','total_purchase']

# Group by user_id, age_group, has_children, and gender and calculate the sum for each group
kmeans_aggregated_df = df.groupby(['user_id'])[columns_to_aggregate].sum().reset_index()

# The resulting DataFrame 'aggregated_data' will have user_id, age_group, has_children, gender,and the summation of the specified columns for each combination of those variables.


# # Data Segmentation

# ## 1) Categorization

# In[33]:


def segment_customer_to_perk(aggregated_data):
    # Condition 1: 1 Night Free Hotel with Flight
    if (
        aggregated_data['hotel_booked'] > 0
        and aggregated_data['flight_booked'] > 0
        and aggregated_data['total_purchase'] > avrg_total_purchase
    ):
        return "1 Night Free Hotel with Flight"

    # Condition 2: Exclusive Discounts
    elif (
        aggregated_data['page_clicks'] > average_page_clicks
        and (
            aggregated_data['hotel_discount_usd'] > avg_hotel_discount_usd
            or aggregated_data['flight_discount_usd'] > avrg_flight_discount_usd
        )
    ):
        return "Exclusive Discounts"

    # Condition 3: Free Hotel Meal
    elif aggregated_data['hotel_booked'] > aggregated_data['flight_booked']:
        return "Free Hotel Meal"

    # Condition 4: Free Checked Bag
    elif aggregated_data['flight_checked_bags'] > avrg_checked_bag:
        return "Free Checked Bag"

    # Condition 5: No Cancellation Fee (Default)
    else:
        return "No Cancellation Fee"

# Apply the segmentation function to your dataset and create a new column
aggregated_data['assigned_perk'] = aggregated_data.apply(segment_customer_to_perk, axis=1)


# In[34]:


# Assuming 'assigned_perk' is a categorical column in your DataFrame
perk_counts = aggregated_data['assigned_perk'].value_counts()

# Create a bar plot
plt.figure(figsize=(8, 6))
perk_counts.plot(kind='bar')
plt.title('Assigned Perks Frequency')
plt.xlabel('Perk')
plt.ylabel('Frequency')
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.show()


# ## 2) Demographic Segmentation

# This segmentation is based on demographic factors such as age group, gender, marital status, and whether they have children.

# In[35]:


def create_segment(age_group, has_children, gender):
    age_labels = ['Under 18', '18-25', '26-35', '36-45', '46-55', '56+']

    if age_group == age_labels[0]:
        if has_children == 0 and gender == 'M':
            return 'Teenage Single Males'
        elif has_children == 0 and gender == 'F':
            return 'Teenage Single Females'
        elif has_children == 0 and gender == 'O':
            return 'Teenage Single Others'
        elif has_children == 1 and gender == 'M':
            return 'Teenage Family Males'
        elif has_children == 1 and gender == 'F':
            return 'Teenage Family Females'
        elif has_children == 1 and gender == 'O':
            return 'Teenage Family Others'
    
    elif age_group == age_labels[1]:
        if has_children == 0 and gender == 'M':
            return 'Young Adult Single Males'
        elif has_children == 0 and gender == 'F':
            return 'Young Adult Single Females'
        elif has_children == 0 and gender == 'O':
            return 'Young Adult Single Others'
        elif has_children == 1 and gender == 'M':
            return 'Young Adult Family Males'
        elif has_children == 1 and gender == 'F':
            return 'Young Adult Family Females'
        elif has_children == 1 and gender == 'O':
            return 'Young Adult Family Others'
    
    elif age_group == age_labels[2]:
        if has_children == 0 and gender == 'M':
            return 'Adult Single Males'
        elif has_children == 0 and gender == 'F':
            return 'Adult Single Females'
        elif has_children == 0 and gender == 'O':
            return 'Adult Single Others'
        elif has_children == 1 and gender == 'M':
            return 'Adult Family Males'
        elif has_children == 1 and gender == 'F':
            return 'Adult Family Females'
        elif has_children == 1 and gender == 'O':
            return 'Adult Family Others'
    
    elif age_group == age_labels[3]:
        if has_children == 0 and gender == 'M':
            return 'Middle-Aged Single Males'
        elif has_children == 0 and gender == 'F':
            return 'Middle-Aged Single Females'
        elif has_children == 0 and gender == 'O':
            return 'Middle-Aged Single Others'
        elif has_children == 1 and gender == 'M':
            return 'Middle-Aged Family Males'
        elif has_children == 1 and gender == 'F':
            return 'Middle-Aged Family Females'
        elif has_children == 1 and gender == 'O':
            return 'Middle-Aged Family Others'
    
    elif age_group == age_labels[4]:
        if has_children == 0 and gender == 'M':
            return 'Senior Single Males'
        elif has_children == 0 and gender == 'F':
            return 'Senior Single Females'
        elif has_children == 0 and gender == 'O':
            return 'Senior Single Others'
        elif has_children == 1 and gender == 'M':
            return 'Senior Family Males'
        elif has_children == 1 and gender == 'F':
            return 'Senior Family Females'
        elif has_children == 1 and gender == 'O':
            return 'Senior Family Others'
    
    elif age_group == age_labels[5]:
        if has_children == 0 and gender == 'M':
            return 'Elderly Single Males'
        elif has_children == 0 and gender == 'F':
            return 'Elderly Single Females'
        elif has_children == 0 and gender == 'O':
            return 'Elderly Single Others'
        elif has_children == 1 and gender == 'M':
            return 'Elderly Family Males'
        elif has_children == 1 and gender == 'F':
            return 'Elderly Family Females'
        elif has_children == 1 and gender == 'O':
            return 'Elderly Family Others'
    
    return 'Other'

# Apply the function to create the 'demographic_segment' column
aggregated_data['demographic_segment'] = aggregated_data.apply(lambda row: create_segment(row['age_group'], row['has_children'], row['gender']), axis=1)


# In[36]:


aggregated_data


# ## 3) K-means Clustering

# # Data Scaling

# Our data doesn't follow a normal distribution.So, Min-Max scaling or robust scaling might be more appropriate than Z-score scaling and robust scaling is a suitable choice when dealing with datasets that contain outliers.

# In[37]:


from sklearn.preprocessing import RobustScaler

columns_to_scale_aggregated= ['flight_discount_usd', 'hotel_discount_usd', 'page_clicks',
                      'flight_base_fare_usd', 'hotel_per_room_usd','total_purchase']

# Create a RobustScaler instance
scaler = RobustScaler()

# Fit and transform the data
kmeans_aggregated_df[columns_to_scale_aggregated] = scaler.fit_transform(kmeans_aggregated_df[columns_to_scale_aggregated])


# In[38]:


kmeans_aggregated_df


# we want to use some of features and behavioral metrics for K-means clustering.
# (We will not use features like 'married' and 'has_children', because Using boolean variables in PCA and K-means clustering can be challenging.
# In fact, PCA and K-means are designed to work with continuous numerical data)

# In[39]:


kmeans_aggregated_df['flight_base_fare_usd'] = kmeans_aggregated_df['flight_base_fare_usd'].fillna(0)
kmeans_aggregated_df['hotel_per_room_usd'] = kmeans_aggregated_df['hotel_per_room_usd'].fillna(0)


# In[40]:


selected_variables = ['flight_discount_usd', 'hotel_discount_usd', 'page_clicks',
                      'flight_base_fare_usd', 'hotel_per_room_usd']
selected_corr_matrix = kmeans_aggregated_df[selected_variables].corr()
sns.heatmap(selected_corr_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap of Selected Variables')
plt.show()


# There is no significant correlation between our feaures , so we can use all of them for PCA.

# # Explained Variance

# It measures how much of the total variance in the data is explained by each PCA component. When we set a threshold (e.g., 90%), we are looking for the smallest number of components that capture at least that percentage of the total variance. This approach aims to reduce dimensionality while retaining a certain amount of information.

# In[41]:


from sklearn.preprocessing import RobustScaler

# Select the features you want to include in PCA
features_to_include = ['flight_base_fare_usd', 'hotel_per_room_usd', 'page_clicks',
                       'flight_discount_usd', 'hotel_discount_usd']

# Extract the data for PCA
data_for_pca = kmeans_aggregated_df[features_to_include]


# Initialize PCA
pca = PCA()

# Fit PCA to the scaled data
pca.fit(data_for_pca)

# Calculate explained variance for each component
explained_variance = pca.explained_variance_ratio_

# Plot explained variance per component
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), marker='o', linestyle='--')
plt.title('Explained Variance vs. Number of PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.grid()
plt.show()


# In[42]:


behavioral_metrics_df = kmeans_aggregated_df[['flight_base_fare_usd', 'hotel_per_room_usd',
                             'page_clicks','flight_discount_usd', 'hotel_discount_usd']]

# Define a threshold for explained variance
explained_variance_threshold = 0.95

# Initialize PCA
pca = PCA()

# Fit PCA to the scaled data
pca.fit(behavioral_metrics_df)

# Find the number of components that explain at least the threshold
num_components_threshold = sum(pca.explained_variance_ratio_.cumsum() < explained_variance_threshold) + 1

# Define a range of PCA components to try
num_components_to_try = [2, 3, 4, 5]

best_num_components = None
best_explained_variance = 0

for n_components in num_components_to_try:
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(behavioral_metrics_df)

    # Calculate the explained variance ratio
    explained_variance = pca.explained_variance_ratio_.sum()

    # Check if this configuration is the best so far
    if explained_variance > best_explained_variance:
        best_explained_variance = explained_variance
        best_num_components = n_components

print(f"The number of PCA components to explain at least {explained_variance_threshold:.0%} variance is: {num_components_threshold}")
print(f"The best number of PCA components is: {best_num_components}")


# In[43]:


# Combine all the behavioral metrics into a new DataFrame
behavioral_metrics_df = kmeans_aggregated_df[['flight_base_fare_usd', 'hotel_per_room_usd',
                            'page_clicks','flight_discount_usd', 'hotel_discount_usd']]

# Apply PCA to reduce dimensionality
pca = PCA(n_components=3)  # Use 6 PCA components
pca_result = pca.fit_transform(behavioral_metrics_df)

# Add PCA results as new columns in the original DataFrame
for i in range(3):
    kmeans_aggregated_df[f'PCA_Component_{i+1}'] = pca_result[:, i]

# K-means clustering with PCA components
X = kmeans_aggregated_df[[f'PCA_Component_{i+1}' for i in range(3)]]  # Use all 6 PCA components
kmeans = KMeans(n_clusters=5, random_state=0, n_init=10)
kmeans.fit(X)
kmeans_aggregated_df['kmeans_cluster'] = kmeans.labels_

# Extract the data and labels from your DataFrame
data = kmeans_aggregated_df[[f'PCA_Component_{i+1}' for i in range(3)]].values
labels = kmeans_aggregated_df['kmeans_cluster'].values

# Get the cluster centroids from your K-means model
centroids = kmeans.cluster_centers_

# Create the scatter plot
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='rainbow')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='black')
plt.xlabel('PCA_Component_1')
plt.ylabel('PCA_Component_2')
plt.title('K-means Clustering with Centroids')
plt.show()


# In[44]:


from mpl_toolkits.mplot3d import Axes3D

# Create the scatter plot with 3 PCA components
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Use the first 3 PCA components for the scatter plot
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='rainbow', s=50)
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='x', s=200, linewidths=3, color='black')

ax.set_xlabel('PCA_Component_1')
ax.set_ylabel('PCA_Component_2')
ax.set_zlabel('PCA_Component_3')
ax.set_title('K-means Clustering with Centroids (3D Scatter Plot)')

plt.show()


# In[45]:


import seaborn as sns

# Combine PCA components and labels into a DataFrame
pca_df = pd.DataFrame(data=pca_result, columns=[f'PCA_Component_{i+1}' for i in range(3)])
pca_df['kmeans_cluster'] = labels

# Create a pair plot
sns.set(style="ticks")
sns.pairplot(pca_df, hue="kmeans_cluster", palette='rainbow')
plt.suptitle("Pair Plot of PCA Components")
plt.show()


# In[46]:


# the list of features we want to include in the pair plot
features_to_include = ['flight_discount_usd', 'hotel_discount_usd',
                       'flight_base_fare_usd', 'hotel_per_room_usd',
                       'page_clicks','user_id']

# The style of the pair plot
sns.set(style='ticks')

# Create the pair plot with hue set to 'kmeans_cluster'
sns.pairplot(kmeans_aggregated_df, hue='kmeans_cluster', vars=features_to_include, palette='Set1')

plt.show()


# In[47]:


# the list of features we want to include in the pair plot
features_to_include = ['flight_discount_usd', 'hotel_discount_usd']

# The style of the pair plot
sns.set(style='ticks')

# Create the pair plot with hue set to 'kmeans_cluster'
sns.pairplot(kmeans_aggregated_df, hue='kmeans_cluster', vars=features_to_include, palette='Set1')

plt.show()


# In[48]:


# Select the 4 columns you want to keep
selected_columns1= aggregated_data[['user_id', 'demographic_segment', 'assigned_perk']]
selected_columns2= kmeans_aggregated_df[['user_id', 'kmeans_cluster']]

# Save the DataFrame as a CSV file
selected_columns1.to_csv('TravelTide_segmentation.csv', index=False)
selected_columns2.to_csv('TravelTide_kmeans.csv', index=False)

