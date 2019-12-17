#데이터 전처리, 정규화 : StandardScaler
import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
           [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
           [20000,30000,40000], [30000,40000,50000], [40000,50000,60000],[100,200,300]]) #(14,3)

y = np.array([4,5,6,7,8,9,10,11,12,13,50000,60000,70000,400]) #(14,)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
scaler = RobustScaler()
scaler.fit(x)
x = scaler.transform(x) 
print(x)

# x_train 첫번째부터 13번째까지
# x_predict 14번째값

x_train = x[:13] 
x_predict = x[13:]

y1 = y[:13]
y2 = y[13:]

#2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(3,)))
model.add(Dense(1000)) 
model.add(Dense(1000)) 
model.add(Dense(10)) 
model.add(Dense(1))

#3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x_train,y1, epochs=1000, verbose=1) #fit에는 잘랐을 때 표본값을 넣어준다.

import numpy as np
# 평가, 예측
yhat = model.predict(x_predict) #테스트값을 넣었을 때 예상값
print(yhat)