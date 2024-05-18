import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
print("libraries and modules loaded")

print("now loading the dataset")

# loding the dataset
x_train = np.loadtxt('input.csv', delimiter=',')
y_train = np.loadtxt('labels.csv', delimiter=',')

x_test = np.loadtxt('input_test.csv', delimiter=',')
y_test = np.loadtxt('labels_test.csv', delimiter=',')


print("looking at its shape")
print("shape of x_train:", x_train.shape)
print("shape of x_train:", y_train.shape)
print("shape of x_train:", x_test.shape)
print("shape of x_train:", y_test.shape)

print("reshaping it")
x_train = x_train.reshape(len(x_train), 100, 100, 3)
y_train = y_train.reshape(len(y_train), 1)
x_test = x_test.reshape(len(x_test), 100, 100, 3)
y_test = y_test.reshape(len(y_test), 1)

print("data divided by 225")
x_train = x_train/255.0
x_test = x_test/255.0

print("again looking at its shape")
print("shape of x_train:", x_train.shape)
print("shape of x_train:", y_train.shape)
print("shape of x_train:", x_test.shape)
print("shape of x_train:", y_test.shape)


print("viewing a random picture")
idx = random.randint(0, len(x_train))
plt.imshow(x_train[idx,:])
plt.show()

print("making the CNN")
# sequential means that the layers will be stacked up 
model = Sequential([
    Conv2D(32,(3,3), activation='relu', input_shape=(100, 100 ,3)),
    MaxPooling2D((2,2)),
    Conv2D(32,(3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64,activation='relu'),
    Dense(1,activation='sigmoid')
])


print("compiling the model and adding cost function and back propagation")
# add the cost function and back propagation 
model.compile(loss= 'binary_crossentropy',optimizer ='adam',metrics =['accuracy'])


print("fitting the model")
model.fit(x_train, y_train, epochs=5, batch_size=64)

print("evalluating the model")
model.evaluate(x_test, y_test)


print("predicting whethr its a cat or a dog")
idx2 = random.randint(0, len(y_test))
plt.imshow(x_test[idx2, :])
plt.show()

y_pred = model.predict(x_test[idx2, :].reshape(1,100,100,3))
y_pred = y_pred > 0.5

if(y_pred == 0 ):
    pred = 'dog'
else:
    pred = 'cat'

print("our model predicted it as a :", pred)