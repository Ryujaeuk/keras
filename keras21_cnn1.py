#cnn의 특징 : 특징을 잡는다. (feature)
from keras.models import Sequential

filter_size = 32
kernel_size = (3,3)

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
model = Sequential()
model.add(Conv2D(7, (2,2), #padding='same', #conv : 특징을 추출, (2,2) : 가로2,세로2로 짜르겠다 => 2,2로 자르면 가로,세로 1개씩 줄어듬 (27,27,7(output))
                 input_shape = (28, 28, 1))) # 가로28/세로28/feature흑백1 , 컬러:3 / (None,10,10,1)
# model.add(Conv2D(16, (2,2)))
# model.add(MaxPooling2D(3,3))
# model.add(Conv2D(8, (2,2)))
model.add(Flatten()) # 27x27x7 = 5103 
model.add(Dense(1))

model.summary()