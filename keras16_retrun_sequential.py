from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
           [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
           [20,30,40], [30,40,50], [40,50,60]])

y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print("x.shape : ", x.shape)
print("y.shape : ", y.shape)

x = x.reshape((x.shape[0], x.shape[1], 1))
print("x.shape : ", x.shape)

#2. 모델 구성
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1), # input (n,3,1) output (n,10)
               return_sequences=True))
model.add(LSTM(10, activation='relu', return_sequences=True))
model.add(LSTM(10, activation='relu', return_sequences=True))
model.add(LSTM(10, activation='relu', return_sequences=True))
model.add(LSTM(10, activation='relu', return_sequences=True))
model.add(LSTM(10, activation='relu', return_sequences=True)) #LSTM을 많이 사용하면 가중치의 문제로 효율성이 떨어진다.
#output이 (n,10)이기 때문에 ndim이 2개밖에 없다고 에러가 뜸
model.add(LSTM(3)) #input(n,10) (error) => retrun_sequences = True 사용 (n, 3, 10)
model.add(Dense(5))
model.add(Dense(1))

model.summary()

'''
#3. 실행
model.compile(optimizer='adam', loss='mse')

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto') 
#model.fit(x, y, epochs=1000, verbose=2)
model.fit(x, y, epochs=1000, callbacks=[early_stopping])

x_input = array([25,35,45])
x_input = x_input.reshape((1,3,1))

yhat = model.predict(x_input)
print(yhat)
'''