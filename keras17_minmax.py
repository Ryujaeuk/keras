#데이터 전처리 => 정규화 : MinMaxScaler = 최소최대스칼라 => 최소, 최대를 0과 1로 설정 (격차가 큰 수들을 0,1로 격차를 줄임) => 
#효과 : 머신의 속도 증가, 과적합 약간의 방지

from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
           [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
           [20000,30000,40000], [30000,40000,50000], [40000,50000,60000]]) #(13,3)

y = array([4,5,6,7,8,9,10,11,12,13,50000,60000,70000,400]) # (13, )

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x) # x를 minmax 시키겠다.
x = scaler.transform(x) # model에서 evaluate, predict하는 과정과 동일 => minmaxscaler를 쓰고 fit, trasform을 해준다.
print(x)


print("x.shape : ", x.shape) # (13,3)
print("y.shape : ", y.shape) # (13, )

#2. 모델 구성
model = Sequential()
# model.add(LSTM(10, activation='relu', input_shape=(3,1))) # (n, 3, 1)
model.add(Dense(10, activation='relu', input_shape=(3,)))
model.add(Dense(5)) #activation(활성화 함수)의 default값 : linear
model.add(Dense(1))

#3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=300, verbose=0)

import numpy as np
# 평가, 예측
x_input = array([25,35,45]) #(3,1)
x_input = np.transpose(x_input) 
x_input = scaler.transform(x_input)

# x_input = x_input.reshape((1,3,1))

yhat = model.predict(x_input)
print(yhat)
