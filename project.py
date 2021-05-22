"""
Title: Data Science Team Project
Class: 2021-1 Data Science
Member: (학번) 배성재
        (학번) 김도균
        201633841 LEE KANG UK
"""

# =========================== Library ===========================
### Step 1. Import the libraries
from datetime import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# =========================== Function ===========================
"""
Name: drop_before_2020
Input: weather.csv dataframe
Output: None
Desc: This function drops what the date value is before 2020.
"""
def drop_before_2020(df):
    indexList = []   # 2020년도가 아닌 data index list
    for key, value in df['date'].iteritems():
        splited_value = value.split('-')
        if splited_value[0] != '2020':
             indexList.append(key)

    for i in indexList:
         df.drop(i, axis=0, inplace=True)   # 2020년도 아닌 data drop

"""
Name: merge_dataframe
Input: TimeProvince dataframe, Weather dataframe and merge method
Output: Return merged dataframe (TimeProvince + Weather)
Desc: This function merges the two dataframes by "date" and "province" columns
"""
def merge_dataframe(time, weather, method):
    new_merged_df = pd.merge(left=time, right=weather, how=method, on=['date', 'province'], sort=False)
    return new_merged_df

# =========================== Main ===========================
# Read the TimeProvince.csv and Weather.csv files
time_province_df = pd.read_csv("./TimeProvince.csv")
weather_df = pd.read_csv("./Weather.csv")

### Step 2. Dataset
print("**** TimeProvince.csv ****")
print(time_province_df.head(), end="\n")
print("**** Weather.csv ****")
print(weather_df.head(), end="\n")

### Step 3. Missing Values
# 결측값 확인 (Weather.csv) --> index 24482 
print("**** Check NaN values of TimeProvince.csv ****")
print(time_province_df.isnull().sum(), end="\n")
print("**** Check NaN values of Weather.csv ****")
print(weather_df.isnull().sum(), end="\n")

# NaN값 채우기? (Delete or fill method) => Drop 하기!
weather_df.dropna(axis=0, inplace=True)

# cf) Weather.csv 파일에서 date가 2020년도가 아닌 data들 drop하기
drop_before_2020(weather_df)

# cf) Weather.csv와 TimeProvince.csv 합치기
new_merged_df = merge_dataframe(time_province_df, weather_df, "inner")

### Step 4. Categorical Value -> Encoding
province = new_merged_df['province']
new_merged_df = new_merged_df.drop("province", axis=1)
lblEncoder = LabelEncoder()
lblEncoder.fit(province)
province_encoded = lblEncoder.transform(province)
new_merged_df["province"] = province_encoded

### Scaling, train/test split?