import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import median

df = pd.read_csv('dataset/cost-of-living.csv')

print(df.head(10))
print(df.shape)
print(df.describe())
print(df.info())

# Check for duplicate city/country combinations
df['city_country'] = df['city'].str.strip() + ', ' + df['country'].str.strip()
print(df.pop('city_country').nunique())

########### Data Cleaning ###########
# Check missing data

# Add a column to indicate missing data then plot a histogram
df['empty_data'] = df.iloc[:, 2:-2].isnull().sum(axis=1)
print(df.isnull().sum())

# Good Data and Poor Data
good_data = df[df['data_quality'] == 1]
poor_data = df[df['data_quality'] == 0]
good_data.drop(['data_quality'], inplace=True, axis=1)
poor_data.drop(['data_quality'], inplace=True, axis=1)

poor_data_missing = poor_data.isnull().sum().iloc[3:]
good_data_missing = good_data.isnull().sum().iloc[3:]

fig = plt.figure(figsize=(12, 4))
plt.title('Raw Dataset Missing Data')
ax1 = plt.subplot(131)
plt.title("Empty data points for poor data")
plt.bar(data=poor_data_missing, x=poor_data_missing.index, height=poor_data_missing)
plt.xticks(rotation=70)

ax2 = plt.subplot(132)
plt.title("Empty data points for good data")
plt.bar(data=good_data_missing, x=good_data_missing.index, height=good_data_missing)
plt.xticks(rotation=70)

# Histogram of Missing Values
ax3 = plt.subplot(133)
step = 1
start = np.floor(min(df['empty_data']) / step) * step
stop = max(df['empty_data']) + step
bin_edges = np.arange(start, stop, step=step)
plt.hist([good_data['empty_data'], poor_data['empty_data']], label=['Good Quality', 'Poor Quality'],
         bins=bin_edges, edgecolor='black', linewidth=0.8, alpha=0.5)
plt.title('Missing Data Count of City-Country Combos')
plt.xlabel('Number of Missing Data Rows')

# Remove rows based on missing data threshold and check for existing missing data
df_new = df[df['empty_data'] < 4]

good_data = df_new[df_new['data_quality'] == 1]
poor_data = df_new[df_new['data_quality'] == 0]
good_data.drop(['data_quality'], inplace=True, axis=1)
poor_data.drop(['data_quality'], inplace=True, axis=1)

poor_data_missing = poor_data.isnull().sum().iloc[3:]
good_data_missing = good_data.isnull().sum().iloc[3:]

fig = plt.figure(figsize=(8, 4))
ax1 = plt.subplot(121)
plt.title("Empty data points for poor data")
plt.bar(data=poor_data_missing, x=poor_data_missing.index, height=poor_data_missing)
plt.xticks(rotation=70)

ax2 = plt.subplot(122)
plt.title("Empty data points for good data")
plt.bar(data=good_data_missing, x=good_data_missing.index, height=good_data_missing)
plt.xticks(rotation=70)

# Cities in each country
num_countries = df.groupby('country').count().iloc[:, 0].sort_values(ascending=False)[0:15]

plt.figure()
plt.barh(data=num_countries, width=num_countries, y=num_countries.index)
plt.show()

# Categorize data columns
restaurants = df_new[df_new.columns[3:11]]
groceries = df_new[df_new.columns[11:26]]
alcohol = df_new[df_new.columns[26:29]]
transit = df_new[df_new.columns[30:35]]
clothes = df_new[df_new.columns[46:50]]
housing = df_new[df_new.columns[50:56]]
salary = df_new[df_new.columns[57]]

categories = [restaurants, groceries, alcohol, transit, clothes, housing, salary]

# Pairplots by Category
# for i, c in enumerate(categories):
#     plot = sns.PairGrid(c)
#     plot.map_diag(plt.hist)
#     plot.map_upper(plt.scatter)
#     plot.map_lower(sns.kdeplot)
#     plt.show()

# Find the strongest variables affecting cost of living
# Median Difference by Country - Difference of the Country's Median Value and the Overall Median Value
std_series = df_new.iloc[:, 3:57].std()

df_median = df_new.drop(columns=['Unnamed: 0', 'city', 'data_quality', 'empty_data']).groupby('country').median()
median_series = pd.DataFrame()

for c in df_median.columns:
    m = median(df_median[c])
    median_series[c] = df_median[c].apply(lambda x: abs(x-m))

fig = plt.figure(figsize=(8, 4))
ax1 = plt.subplot(121)
plt.title("Standard Deviation")
plt.bar(data=std_series, x=std_series.index, height=std_series)
plt.xticks(rotation=70)

ax2 = plt.subplot(122)
plt.title("Median Difference of Category By Country")
sns.heatmap(median_series, annot=True)


# Map the cost of living by country
