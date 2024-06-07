import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data = pd.read_csv("student-data.csv")
data = data.drop(["Empty"],axis=1)



x = data.drop(["GPA"], axis=1)
y = data[["GPA"]]

'''
print(x.loc[x['Reg no'] == 200701107,:].values)

print(x.head())
print(y.head())

model = LinearRegression()
x_train,x_test,y_train,y_tes = train_test_split(x.values,y.values,test_size=0.3,random_state=4)
train = model.fit(x_train,y_train)

pred = train.predict(x_test)

print(r2_score(pred,y_tes))
'''