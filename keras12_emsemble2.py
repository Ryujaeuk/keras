#1. 데이터
import numpy as np
x1 = np.array([range(100), range(311,411), range(100)])
x2 = np.array([range(501,601), range(711,811), range(100)])

y1 = np.array([range(100,200), range(311,411), range(100,200)])
y2 = np.array([range(501,601), range(711,811), range(100)])
y3 = np.array([range(401,501), range(211,311), range(100)])

x1 = np.transpose(x1)
y1 = np.transpose(y1)
x2 = np.transpose(x2)
y2 = np.transpose(y2)
y3 = np.transpose(y3)


print(x1.shape) # (100,3)
print(x2.shape) # (100,3)
print(y1.shape) # (100,3)
print(y2.shape) # (100,3)
print(y3.shape) # (100,3)



from sklearn.model_selection import train_test_split
#train, test 영역 split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, random_state = 33, train_size = 0.4, shuffle=False
)
x1_val, x1_test, y1_val, y1_test = train_test_split(
    x1_test, y1_test, random_state = 33, test_size = 0.5, shuffle=False
)
#train1(4):test(3):val(3)

x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, random_state = 33, train_size = 0.4, shuffle=False
)
x2_val, x2_test, y2_val, y2_test = train_test_split(
    x2_test, y2_test, random_state = 33, test_size = 0.5, shuffle=False
)

y3_train, y3_test = train_test_split(
    y3, random_state = 66, train_size = 0.4
)
y3_val, y3_test = train_test_split(
    y3_test, random_state = 66, test_size = 0.5
)
print(y1_test.shape)
print(y2_test.shape)
print(y3_test.shape)


#2. 모델구성
# emsemble(앙상블) : 순차형 모델보다 함수형 모델 주로 사용

from keras.models import Sequential, Model 
from keras.layers import Dense, Input

input1 = Input(shape=(3,)) 
dense1 = Dense(1, activation='relu')(input1)
dense2 = Dense(2)(dense1) 
dense3 = Dense(3)(dense2) 
middle1 = Dense(3)(dense3) 

input2 = Input(shape=(3,)) 
xx = Dense(3, activation='relu')(input2) 
xx = Dense(2)(xx) 
xx = Dense(1)(xx) 
middle2 = Dense(3)(xx) 

#concatenate

from keras.layers.merge import concatenate
merge1 = concatenate([middle1, middle2]) 
#병합시킬 레이어들의 마지막 output부분을 concatenante()함수 안에 명시
#2개 이상의 input, output일 때 항상 list사용

output1 = Dense(3)(merge1)
output1 = Dense(15)(output1)
output1 = Dense(3)(output1)

output2 = Dense(3)(merge1)
output2 = Dense(15)(output2)
output2 = Dense(3)(output2)

output3 = Dense(3)(merge1)
output3 = Dense(15)(output3)
output3 = Dense(3)(output3)


model = Model(input = [input1, input2], 
              outputs = [output1, output2, output3])
model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', #metrics=['accuracy']
                metrics=['mse'])
model.fit([x1_train, x2_train], [y1_train, y2_train, y3_train], epochs=100, batch_size=1,
        validation_data=([x1_val, x2_val], [y1_val, y2_val, y3_val]))



#4. 평가 예측 
mse = model.evaluate([x1_test, x2_test], [y1_test, y2_test, y3_test], batch_size=1)
print("mse : ", mse)
# model : input1, input2, merge1, output1, output2 => 5개 


y1_predict, y2_predict, y3_predict = model.predict([x1_test, x2_test])
print(y1_predict, y2_predict, y3_predict)

from sklearn.metrics import mean_squared_error
def RMSE(xxx, yyy):
    return np.sqrt(mean_squared_error(xxx,yyy))
RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
RMSE3 = RMSE(y3_test, y3_predict)
#RMSE(표본, 예상)

print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE3 : ", RMSE3)
print("RMSE : ", (RMSE1+RMSE2+RMSE3)/3)

from sklearn.metrics import r2_score
r2_y1_predict = r2_score(y1_test, y1_predict)
r2_y2_predict = r2_score(y2_test, y2_predict)
r2_y3_predict = r2_score(y3_test, y3_predict)
#r2_score(표본, 예상)

print("R2_1 : ", r2_y1_predict)
print("R2_2 : ", r2_y2_predict)
print("R2_3 : ", r2_y3_predict)
print("R2 : ", (r2_y1_predict + r2_y2_predict+r2_y3_predict)/3)