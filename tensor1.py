import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import load_img

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
	               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# print(train_labels)
# print(test_images[0].shape)
train_images = train_images / 255.0
test_images = test_images / 255.0

def train():
	

	model = keras.Sequential([
	    keras.layers.Flatten(input_shape=(28, 28)),
	    keras.layers.Dense(128, activation=tf.nn.relu),
	    keras.layers.Dense(10, activation=tf.nn.softmax)
	])

	model.compile(optimizer=tf.train.AdamOptimizer(), 
	              loss='sparse_categorical_crossentropy',
	              metrics=['accuracy'])

	model.fit(train_images, train_labels, epochs=3)

	test_loss, test_acc = model.evaluate(test_images, test_labels)

	print('Test accuracy:', test_acc)
	model.save('my_model.h5')


def test():
	img_1 = cv2.imread('./images.jpeg', 0)
	img_1= cv2.resize(img_1, (28, 28))
	img_1=img_1/255.0
	cv2.imshow('image',img_1)
	cv2.waitKey(0)
	# print(img_1)
	# img_2=np.reshape(img_1,(1,28,28))
	# print(img_2.shape)
	img = (np.expand_dims(img_1,0))
	print(img.shape)

	model=keras.models.load_model('my_model.h5')
	model.compile(optimizer=tf.train.AdamOptimizer(), 
	              loss='sparse_categorical_crossentropy',
	              metrics=['accuracy'])
	# pred=model.predict(img)
	# print(pred)
	# print(class_names[np.argmax(pred)])
	# print("------------")
	cv2.imshow('image',test_images[2])
	cv2.waitKey(0)
	pred1=model.predict(np.expand_dims(test_images[2],0))
	print(class_names[np.argmax(pred1)])
	print(class_names[test_labels[2]])

if __name__ == '__main__':
	train()
	test()