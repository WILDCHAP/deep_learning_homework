import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers
from keras import models
from keras.optimizers import SGD

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


#定义LeNet模型
network = models.Sequential()
#这里要设置自动填充，不然28×28的不够
network.add(layers.Conv2D(filters=6, kernel_size=(5,5), padding='valid',
                 input_shape=(28, 28, 1), activation='tanh'))
network.add(layers.MaxPooling2D(pool_size=(2,2)))
network.add(layers.Conv2D(filters=16,kernel_size=(5,5),padding='valid',activation='tanh'))
network.add(layers.MaxPooling2D(pool_size=(2,2)))
network.add(layers.Flatten())
network.add(layers.Dense(120,activation='tanh'))
network.add(layers.Dense(84,activation='tanh'))
network.add(layers.Dense(10,activation='softmax'))

#编译网络：优化器(SGD), 损失函数(二分类交叉熵误差), 监控指标(accuracy)
sgd=SGD(lr=0.05,decay=1e-6,momentum=0.9,nesterov=True)
network.compile(optimizer=sgd,
				loss='categorical_crossentropy',
				metrics=['accuracy'])
				
#训练网络：epochs=5, batch_size=128
History = network.fit(train_images, train_labels, epochs=5, batch_size=128)

#用训练好的网络测试测试数据精度
print("该网络在测试数据上的[损失值/准确度]为：", network.evaluate(test_images, test_labels))

#绘制图像
history_dict = History.history
acc_list = history_dict['accuracy']

x = range(1, len(acc_list)+1)

plt.plot(x, acc_list, linestyle=":", label='train_acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('accuracy')
plt.legend()
plt.show()
