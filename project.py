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
print(weather_df['province'].unique())
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


print(new_merged_df)

## Export
# =========================== Main ===========================
# Read the test.csv files

# new_merged_df.drop(['Unnamed: 0'], axis=1, inplace=True)

# 지역코드 따로 빼기 -> 각각 csv파일로 저장시키기
location = new_merged_df['code'].drop_duplicates()

print(len(location))

for i in range(0, len(location)):
    temp = new_merged_df[new_merged_df['code'] == location[i]]
    file_name = "location_" + str(location[i]) + ".csv"

    temp = pd.concat([temp], ignore_index = True)
    temp['target'] = temp['code']

    # Confirmed 구하기
    for j in range(len(temp)-1, 1, -1):
        real = temp.iloc[j,2] - temp.iloc[j-1,2]
        temp.iloc[j,2] = real

    # Released 구하기
    for j in range(len(temp)-1, 1, -1):
        real = temp.iloc[j,3] - temp.iloc[j-1,3]
        temp.iloc[j,3] = real

    # Deceased 구하기
    for j in range(len(temp)-1, 1, -1):
        real = temp.iloc[j,4] - temp.iloc[j-1,4]
        temp.iloc[j,4] = real

    # Target 단계 구하기
    for j in range(len(temp)-1, 0, -1):
        if temp.iloc[j,2] >=  500:
            temp.iloc[j,14] = 7
        elif temp.iloc[j,2] >=  251:
            temp.iloc[j,14] = 6
        elif temp.iloc[j,2] >=  101:
            temp.iloc[j,14] = 5
        elif temp.iloc[j,2] >=  51:
            temp.iloc[j,14] = 4
        elif temp.iloc[j,2] >=  26:
            temp.iloc[j,14] = 3
        elif temp.iloc[j,2] >=  11:
            temp.iloc[j,14] = 2
        elif temp.iloc[j,2] >=  1:
            temp.iloc[j,14] = 1
        elif temp.iloc[j,2] ==  0:
            temp.iloc[j,14] = 0

    # TODO:
    # 각 지역별 비닝한 기준을 보여줄 자료 필요
    # 지역별 Scatter plot 시각화하면 될지도?

    temp.drop(index=0, axis=0, inplace=True)
    temp.to_csv(file_name)

