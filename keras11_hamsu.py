#1. 데이터
import numpy as np
x = np.array(range(1,101))
y = np.array(range(1,101))
print(x)

from sklearn.model_selection import train_test_split
#train, test 영역 split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state = 66, train_size = 0.7, shuffle=False
)
#분할된 test영역에서 test, val 영역 split
x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, random_state = 66, test_size = 0.5, shuffle=False
) #6:2:2


#2. 모델구성
from keras.models import Sequential, Model 
from keras.layers import Dense, Input

#Sequential : 순차형 모델
#model = Sequential()
# model.add(Dense(500, input_dim=1, activation='relu'))

#함수형 모델 특징 : 
#1. 맨 처음에 input을 명시.
#2. 추가되는 layer마다 input을 맨 뒤에 명시해준다.(output)(input)
#3. model의 시작 input과 끝 output을 명시해준다.(model의 범위)
# input1 = Input(shape=(1,)) #1
# dense1 = Dense(5, activation='relu')(input1) #2
# dense2 = Dense(3)(dense1) #2
# dense3 = Dense(4)(dense2) #2
# output1 = Dense(1)(dense8) #2

input1 = Input(shape=(1,)) 
xx = Dense(5, activation='relu')(input1) 
xx = Dense(3)(xx) 
xx = Dense(4)(xx) 
xx = Dense(2)(xx)
xx = Dense(1)(xx)
xx = Dense(2)(xx)
xx = Dense(3)(xx)
xx = Dense(4)(xx)
output1 = Dense(1)(xx) 


model = Model(input = input1, outputs = output1) #3
model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', #metrics=['accuracy']
                metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
        validation_data=(x_val, y_val))

#4. 평가 예측 
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)

y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

# 레이어를 10개 이상 늘리시오.