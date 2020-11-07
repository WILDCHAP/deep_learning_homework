import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#print(train_images.shape)

#预处理：不进行数据增强，但将图像数据平铺展开
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255		#将其压缩到0~1

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255		#将其压缩到0~1

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#定义神经网络：两层全链接层
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))	#隐藏层512个单元
network.add(layers.Dense(10, activation='softmax'))		#输出层10个单元

#编译网络：优化器(rmsprop), 损失函数(二分类交叉熵误差), 监控指标(accuracy)
network.compile(optimizer='rmsprop',
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
