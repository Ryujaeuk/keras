import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1,11))

size = 5
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print("====================")
print(dataset)

x_train = dataset[:,0:4] #[행,열]
y_train = dataset[:,-1]
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

print(x_train.shape) #(6,4)
print(y_train.shape) #(6,)

# lstm으로 수정
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(4,1)))
# model.add(Dense(3333, input_shape=(4, )))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300)

x2 = np.array([[7,8,9,10]]) # (4, ) => (1, 4)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)
# x2 = np.array([7,8,9,10]) # (4, ) => (1, 4)
# x2 = x2.reshape(1,x2.shape,1)
y_pred = model.predict(x2)
print(y_pred)