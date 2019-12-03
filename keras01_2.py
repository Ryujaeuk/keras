#layer의 개수와 dense(노드)개수 수정, 
from keras.models import Sequential
from keras.layers import Dense

import numpy as np
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])
x2 = np.array([11,12,13,14,15])

model = Sequential()
model.add(Dense(100, input_dim=1, activation='relu'))
model.add(Dense(50))
model.add(Dense(75))
model.add(Dense(100))
model.add(Dense(75))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x,y, epochs=100, batch_size=1)

loss, acc = model.evaluate(x, y, batch_size=1)
print("acc : ", acc)
print("loss : ", loss)

y_predict = model.predict(x2)
print(y_predict)