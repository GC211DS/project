"""
Title: Data Science Team Project
Class: 2021-1 Data Science
Member: 201533645 배성재
        201533631 김도균
        201633841 LEE KANG UK
"""

# =========================== Library ===========================
### Step 1. Import the libraries
import pandas as pd
from dateutil.rrule import weekday
from pandas.core.frame import DataFrame
from sklearn.preprocessing import LabelEncoder


# =========================== Function ===========================
"""
Name: drop_before_2020
Input: weather.csv dataframe
Output: None
Desc: This function drops what the date value is before 2020.
"""
def drop_before_2020(df):
    indexList = []   # The list of data what year is not 2020
    for key, value in df['date'].iteritems():
        splited_value = value.split('-')
        if splited_value[0] != '2020':
             indexList.append(key)

    for i in indexList:
         df.drop(i, axis=0, inplace=True)   # Drop data if not in 2020

"""
Name: merge_dataframe
Input: TimeProvince dataframe, Weather dataframe and merge method
Output: Return merged dataframe (TimeProvince + Weather)
Desc: This function merges the two dataframes by "date" and "province" columns
"""
def merge_dataframe(time, weather, method):
    new_merged_df = pd.merge(left=time, right=weather, how=method, on=['date', 'province'], sort=False)
    return new_merged_df



## 1. Data cleaning and merge valuable data
# =========================== Main ===========================

### Step 1-1. Read the TimeProvince.csv and Weather.csv files
time_province_df = pd.read_csv("./TimeProvince.csv")
weather_df = pd.read_csv("./Weather.csv")

### Step 1-2. Dataset
print("**** TimeProvince.csv ****")
print(time_province_df.head(), end="\n")
print("**** Weather.csv ****")
print(weather_df.head(), end="\n")


### Step 1-3. Missing Values (Weather.csv)
## 1-3-1. Check Missing Values --> index 24482 
print("**** Check NaN values of TimeProvince.csv ****")
print(time_province_df.isnull().sum(), end="\n")
print("**** Check NaN values of Weather.csv ****")
print(weather_df.isnull().sum(), end="\n")

## 1-3-2. Fix wrong value (Chunghceongbuk-do -> Chungcheongbuk-do)
weather_df.replace("Chunghceongbuk-do", 'Chungcheongbuk-do', inplace=True)

## 1-3-3. Fill missing values (Delete or fill method) => Drop 
weather_df.dropna(axis=0, inplace=True)
print(weather_df['province'].unique())

## 1-3-4. if date value isn't 2020 in Weather.csv => Drop
drop_before_2020(weather_df)

### Step 1-4. Merge Weather.csv and TimeProvince.csv
new_merged_df = merge_dataframe(time_province_df, weather_df, "inner")
print(time_province_df)
print(weather_df)
print(new_merged_df)


### Step 1-5. Categorical Value -> Encoding
province = new_merged_df['province']
new_merged_df = new_merged_df.drop("province", axis=1)
lblEncoder = LabelEncoder()
lblEncoder.fit(province)
province_encoded = lblEncoder.transform(province)
new_merged_df["province"] = province_encoded

print(new_merged_df.info())
print(new_merged_df.describe())



## 2. Export preprocessed dataframe
# =========================== Main ===========================

# Step 2-1. Export location data to csv file
location = new_merged_df['code'].drop_duplicates()
location_province = weather_df['province'].drop_duplicates().to_numpy()
location_code = weather_df['code'].drop_duplicates().to_numpy()

location_df = DataFrame({
    "code": location_code,
    "label": location_province,
})
location_df.to_csv("location.csv")


# Step 2-2. Separate by province code and save each csv file
for i in range(0, len(location)):
    temp = new_merged_df[new_merged_df['code'] == location[i]]
    file_name = "location_" + str(location[i]) + ".csv"

    temp = pd.concat([temp], ignore_index = True)
    temp['target'] = temp['code']
    
    ### 2-2-1. Change cumulative value to real value
    # Confirmed
    for j in range(len(temp)-1, 1, -1):
        real = temp.iloc[j,2] - temp.iloc[j-1,2]
        temp.iloc[j,2] = real

    # Released
    for j in range(len(temp)-1, 1, -1):
        real = temp.iloc[j,3] - temp.iloc[j-1,3]
        temp.iloc[j,3] = real

    # Deceased
    for j in range(len(temp)-1, 1, -1):
        real = temp.iloc[j,4] - temp.iloc[j-1,4]
        temp.iloc[j,4] = real

    ### Step 2-2-2. Binning target(confirmed) data
    for j in range(len(temp)-1, 0, -1):
        if temp.iloc[j,2] >=  500:
            temp.iloc[j,14] = 7
        elif temp.iloc[j,2] >=  250:
            temp.iloc[j,14] = 6
        elif temp.iloc[j,2] >=  100:
            temp.iloc[j,14] = 5
        elif temp.iloc[j,2] >=  50:
            temp.iloc[j,14] = 4
        elif temp.iloc[j,2] >=  25:
            temp.iloc[j,14] = 3
        elif temp.iloc[j,2] >=  10:
            temp.iloc[j,14] = 2
        elif temp.iloc[j,2] >=  1:
            temp.iloc[j,14] = 1
        elif temp.iloc[j,2] ==  0:
            temp.iloc[j,14] = 0
    
    # First value can't calculate real value because raw data was cumulative.
    # So, drop useless row (first row)
    temp.drop(index=0, axis=0, inplace=True)

    ### Step 2-2-3. Save to csv file
    temp.to_csv(file_name)

