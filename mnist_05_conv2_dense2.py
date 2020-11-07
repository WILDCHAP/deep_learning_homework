import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#将数据转换为卷积网络形式(1个通道)
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))
#one-hot表示
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255


#将标签标量化(例如第一个数字是3, 那么就转换成0 0 0 1 0 0 0 0 0 0)
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

dev_images = test_images[:8000]
dev_labels = test_labels[:8000]

#定义神经网络：四层卷积层接展开层链接两层全链接层
network = models.Sequential()
#第一层：32个卷积核，每个大小3*3，池化2*2
network.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
network.add(layers.MaxPooling2D((2, 2)))
#第二层：64个卷积核，每个大小3*3，池化2*2
network.add(layers.Conv2D(64, (3,3), activation='relu'))
network.add(layers.MaxPooling2D((2, 2)))
'''#第三层：64个卷积核，每个大小3*3，池化2*2
network.add(layers.Conv2D(64, (3,3), padding='valid', activation='relu'))
network.add(layers.MaxPooling2D((2, 2)))
#第三层：64个卷积核，每个大小3*3，池化2*2
network.add(layers.Conv2D(64, (3, 3), activation='relu'))
network.add(layers.MaxPooling2D((2, 2)))
#第四层：64个卷积核，每个大小3*3，池化2*2
network.add(layers.Conv2D(64, (3, 3), activation='relu'))
network.add(layers.MaxPooling2D((2, 2)))'''
#展开平铺
network.add(layers.Flatten())
#第五层：128个隐藏层
network.add(layers.Dense(128, activation='relu'))
network.add(layers.Dropout(0.5))
#第六层：10个输出
network.add(layers.Dense(10, activation='softmax'))

#编译网络：优化器(rmsprop), 损失函数(二分类交叉熵误差), 监控指标(accuracy)
network.compile(optimizer='rmsprop',
				loss='categorical_crossentropy',
				metrics=['accuracy'])
				
#训练网络：epochs=5, batch_size=128
History = network.fit(train_images, train_labels, epochs=10, batch_size=256, validation_data=(dev_images, dev_labels))

#用训练好的网络测试测试数据精度
print("该网络在测试数据上的[损失值/准确度]为：", network.evaluate(test_images, test_labels))

#绘制图像
history_dict = History.history
acc_list = history_dict['accuracy']
acc_list_2 = history_dict['val_accuracy']

x = range(1, len(acc_list)+1)

plt.plot(x, acc_list, linestyle="-", label='train_acc')
plt.plot(x, acc_list_2, linestyle=":", label='test_acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('accuracy')
plt.legend()
plt.show()

