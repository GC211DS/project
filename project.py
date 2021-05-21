"""
Title: Data Science Team Project
Class: 2021-1 Data Science
Member: (학번) 배성재
        (학번) 김도균
        201633841 LEE KANG UK
"""

# =========================== Library ===========================
### Step 1. Import the libraries
import pandas as pd

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
# 결측값 확인 --> index 24482 
print("**** Check NaN values of TimeProvince.csv ****")
print(time_province_df.isnull().sum(), end="\n")
print("**** Check NaN values of Weather.csv ****")
print(weather_df.isnull().sum(), end="\n")

# NaN값 채우기? (Delete or fill method) => Drop 하기!

# cf) Weather.csv 파일에서 date가 2020년도가 아닌 data들 drop하기
drop_before_2020(weather_df)

# cf) Weather.csv와 TimeProvince.csv 합치기