#test, train data 분리하는 이유
#train데이터로 '최적의 w'값을 구하고 train 데이터와 다른 test데이터로 '최적의 w'값을 검증(evaluate)을 한다.
#x_test = x_predict

from keras.models import Sequential
from keras.layers import Dense

import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])

model = Sequential()
model.add(Dense(500, input_dim=1, activation='relu'))
model.add(Dense(300))
model.add(Dense(1000))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100)

loss, acc = model.evaluate(x_test, y_test)
print("acc : ", acc)
print("loss : ", loss)

y_predict = model.predict(x_test)
print(y_predict)