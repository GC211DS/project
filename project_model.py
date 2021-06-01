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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns


# =========================== Function ===========================
"""
Name: drop_before_2020
Input: weather.csv dataframe
Output: None
Desc: This function drops what the date value is before 2020.
"""
def linear(x,y):
    linearReg = LinearRegression()
    linearReg.fit(x, y)
    return linearReg

"""
Name: coefplot
Input: coef, column
Output: plt.show
Desc: This function plot the coef with column name
"""
def coefplot(results,column):
    plt.title("Coef about column")
    plt.ylabel("Coef")
    plt.scatter(column, results, alpha=1, label="predict")
    return plt.show()

"""
Name: scatterplot
Input: y_test, predict by fitted model, dataset
Output: None
Desc: This function plot scatter plot about data
"""
def scatterplot(y,predict,data):

    plt.scatter(data[:, 6], predict, alpha=0.4, label="predict")
    plt.scatter(data[:, 6], y, alpha=0.4, label="Test data")
    plt.legend()
    plt.xlabel("precipitation")
    plt.ylabel("Confirmed")
    plt.title("Confirmed & Precipitation")
    plt.show()



    plt.scatter(data[:, 5], predict, alpha=0.4, label="predict")
    plt.scatter(data[:, 5], y, alpha=0.4, label="Test data")
    plt.legend()
    plt.xlabel("Max_temp")
    plt.ylabel("Confirmed")
    plt.title("Confirmed & Max_temp")
    plt.show()

    plt.scatter(data[:, 1], predict, alpha=0.4, label="predict")
    plt.scatter(data[:, 1], y, alpha=0.4, label="Test data")
    plt.legend()
    plt.xlabel("deceased")
    plt.ylabel("Confirmed")
    plt.title("Confirmed & deceased")
    plt.show()
    return



# =========================== Main ===========================
# Read the test.csv files
df = pd.read_csv("./test.csv")

df.drop(['Unnamed: 0'], axis=1, inplace=True)

# 지역코드 따로 빼기 -> 각각 csv파일로 저장시키기
location = df['code'].drop_duplicates()

for i in range(0, len(location)):
    temp = df[df['code'] == location[i]]
    file_name = "location_" + str(location[i]) + ".csv"
    temp.to_csv(file_name)

# 지역별로 나눴던 csv 파일들을 dataframe으로 변환
df_10000 = pd.read_csv("location_10000.csv")
df_11000 = pd.read_csv("location_11000.csv")
df_12000 = pd.read_csv("location_12000.csv")
df_13000 = pd.read_csv("location_13000.csv")
df_14000 = pd.read_csv("location_14000.csv")
df_15000 = pd.read_csv("location_15000.csv")
df_16000 = pd.read_csv("location_16000.csv")
df_20000 = pd.read_csv("location_20000.csv")
df_30000 = pd.read_csv("location_30000.csv")
df_40000 = pd.read_csv("location_40000.csv")
df_41000 = pd.read_csv("location_41000.csv")
df_50000 = pd.read_csv("location_50000.csv")
df_51000 = pd.read_csv("location_51000.csv")
df_60000 = pd.read_csv("location_60000.csv")
df_61000 = pd.read_csv("location_61000.csv")
df_70000 = pd.read_csv("location_70000.csv")
    
#scaling & drop data
standardScaler = StandardScaler()
# standardScaler.fit(df.drop(['confirmed','date','time'],axis = 1))
x_scaled = standardScaler.fit_transform(df_10000.drop(['confirmed','date','time'],axis = 1))
#x_scaled = df.drop(['confirmed','date','Unnamed: 0','time'],axis = 1)
columns = df_10000.drop(['confirmed','date','time'],axis = 1).columns

df_scaled_10000 = pd.DataFrame(x_scaled, columns=columns)

# split data set
x_train, x_test, y_train, y_test = train_test_split(df_scaled_10000, df['confirmed'], test_size=0.2, shuffle=True, random_state=34)
# linear regression
model = LinearRegression()
model.fit(x_train, y_train)
print("linear regression score : " ,model.score(x_test,y_test))
print(model.coef_)



# 1. 지역별로 테이블 분해




    # 지역 내 누적값.
    # 지역별로 Preprocessing 다시 해야 함.

    # # Heatmap code
    # corrMat = dft.corr()
    # feature = corrMat.index
    # ax = plt.axes()
    # ax.set_title(location[i])
    
    # plt.figure(figsize=(11,11))
    # g = sns.heatmap(dft[feature].corr(), annot=True, cmap="RdYlGn", ax = ax)
    
    # plt.show()

    # # 2. 온도를 비교하여 분석

    # # split data set
    # x_train, x_test, y_train, y_test = train_test_split(dft, df['confirmed'], test_size=0.2, shuffle=True, random_state=34)

    # #linear regression
    # model = LinearRegression()
    # model.fit(x_train, y_train)
    # print("linear regression score : " ,model.score(x_test,y_test))
    # print(model.coef_)

    # coefplot(model.coef_, df.drop(['confirmed','date','Unnamed: 0','time'],axis = 1).columns)
    # scatterplot(y_test,model.predict(x_test),x_test)



