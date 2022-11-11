
# coding: utf-8

#资料预处理
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
np.random.seed(10)

(x_train, y_train),(x_test,y_test) = mnist.load_data() #读取MNIST资料

#将features（数字影像特征值）以reshape转换为6000*28*28*1的4维矩阵
x_train4D = x_train.reshape(x_train.shape[0],28,28,1).astype('float32')
x_test4D = x_test.reshape(x_test.shape[0],28,28,1).astype('float32')

#将features标准化，可以提高模型预测的准确度，并且更快收敛
x_train4D_normalize = x_train4D / 255
x_test4D_normalize = x_test4D / 255

#使用np_utils.to_categorical, 将训练资料集与测试的label,进行 Onehot encoding 转换
y_trainOneHot = np_utils.to_categorical(y_train)
y_testOneHot = np_utils.to_categorical(y_test)

#建立模型
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D

#建立keras的Sequential模型（线性堆积模型），后续只需要使用model.add()方法，将各神经网络层加入模型即可
model = Sequential()

#建立卷积层1.
#输入的数字影像是28*28大小，执行第一次卷积运算，会产生16个影像，卷积运算并不会改变影像大小，所以仍然是28*28大小。
model.add(Conv2D(filters=16,
                kernel_size=(5,5),#卷积核的大小
                padding='same',#补零
                input_shape=(28,28,1),
                activation='relu'))#激活方式这种算起来比较快

#建立池化层
#输入参数pool_size=(2,2),执行第一次缩减取样，将16个28*28影像，缩小为16个14*14的影像。
model.add(MaxPooling2D(pool_size=(2,2)))

#建立卷积层2.
#输入的数字影像是28*28大小，执行第2次卷积运算，将原本16个的影像，转换为36个影像，卷积运算并不会改变影像大小，所以仍然是14*14大小。
model.add(Conv2D(filters=36,
                kernel_size=(5,5),
                padding='same',#补零
                activation='relu'))

#建立池化层2
#输入参数pool_size=(2,2),执行第2次缩减取样，将36个14*14影像，缩小为36个7*7的影像。
model.add(MaxPooling2D(pool_size=(2,2)))

#加入Dropout(0.25)层至模型中。其功能是，每次训练迭代时，会随机的在神经网络中放弃25%的神经元，以避免overfitting。
model.add(Dropout(0.25))

#建立平坦层
#之前的步骤已经建立池化层2，共有36个7*7影像，转换为1维的向量，长度是36*7*7=1764，也就是1764个float数字，正好对应到1764个神经元。
model.add(Flatten())#铺开，变成一维的

#建立隐藏层，共有128个神经元
model.add(Dense(128,activation='relu'))

#加入Dropout(0.5)层至模型中。其功能是，每次训练迭代时，会随机的在神经网络中放弃50%的神经元，以避免overfitting。
model.add(Dropout(0.5))

#建立输出层
#共有10个神经元，对应到0-9共10个数字。并且使用softmax激活函数进行转换，softmax可以将神经元的输出，转换为预测每一个数字的几率。
model.add(Dense(10,activation='softmax'))

#查看模型的摘要
print(model.summary())

#进行训练
#定义训练方式
#在模型训练之前，我们必须使用compile方法，对训练模型进行设定
model.compile(loss='categorical_crossentropy',
             optimizer='adam',metrics=['accuracy'])

#开始训练
train_history=model.fit(x=x_train4D_normalize,
                       y=y_trainOneHot,validation_split=0.2,#20%用于交叉验证
                       epochs=10,batch_size=300,verbose=2)

#画出accuracy执行结果
#之前训练步骤，会将每一个训练周期的accuracy与loss，记录在train_history。
import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()

show_train_history(train_history, 'accuracy','val_accuracy')

#画出loss误差执行结果
show_train_history(train_history,'loss','val_loss')

#用test评估模型准确度
scores = model.evaluate(x_test4D_normalize,y_testOneHot)
scores[1]

#进行预测 
prediction=model.predict_classes(x_test4D_normalize)

#预测结果
prediction[:10]

#显示前10笔预测结果
def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num>25:
        num =25
    for i in range(0,num):
        ax = plt.subplot(5,5,1+i)
        ax.imshow(images[idx], cmap='binary')
        title = "label="+str(labels[idx])
        if len(prediction)>0:
            title+=",predict="+str(prediction[idx])
        ax.set_title(title,fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx+=1
    plt.show()

plot_images_labels_prediction(x_test,y_test,prediction,idx=0)

#显示混淆矩阵
import pandas as pd
pd.crosstab(y_test,prediction,rownames=['label'],colnames=['predict'])
