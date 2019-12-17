#x, y 데이터를 각각 2개씩으로 분리
#2개의 인풋, 2개의 아웃풋 모델인 ensemble모델 구현 

from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
           [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
           [20,30,40], [30,40,50], [40,50,60]])

y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

# print("x.shape : ", x.shape)
# print("y.shape : ", y.shape)

# x = x.reshape((x.shape[0], x.shape[1], 1))
# print("x.shape : ", x.shape)

x1 = x[:10]
x2 = x[10:]
print("x1.shape : ", x1.shape)
print("x2.shape : ", x2.shape)
print(x1)
print(x2)

y1 = y[:10]
y2 = y[10:]
print("y1.shape : ", y1.shape)
print("y2.shape : ", y2.shape)
print(y1)
print(y2)
#2. 모델 구성
# model = Sequential()
# model.add(LSTM(10, activation='relu', input_shape=(3,1)))
# model.add(Dense(1000))
# model.add(Dense(1000))
# model.add(Dense(100))
# model.add(Dense(1))

input1 = Input(shape=(3,1)) 
xx = LSTM(3, activation='relu')(input1)
xx = Dense(2)(xx) 
xx = Dense(3)(xx) 
middle1 = Dense(3)(xx) 

input2 = Input(shape=(3,1)) 
xx = LSTM(3, activation='relu')(input2) 
xx = Dense(2)(xx) 
xx = Dense(1)(xx) 
middle2 = Dense(3)(xx) 

from keras.layers.merge import concatenate
merge1 = concatenate([middle1, middle2]) 

output1 = Dense(30)(merge1)
output1 = Dense(15)(output1)
output1 = Dense(10,)(output1)

output2 = Dense(15)(merge1)
output2 = Dense(15)(output2)
output2 = Dense(3,)(output2)

model = Model(input = [input1, input2], outputs = [output1, output2])
model.summary()
#3. 실행
model.compile(optimizer='adam', loss='mse')

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto') 
#model.fit(x, y, epochs=1000, verbose=2)
model.fit([x1, x2],[y1, y2], epochs=1000, callbacks=[early_stopping])

x_input = array([25,35,45])
x_input = x_input.reshape((10,3,1))

yhat = model.predict(x_input)
print(yhat)