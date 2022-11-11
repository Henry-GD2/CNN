# CNN
基于卷积神经网络的手写数字识别


主要是用来进行分类任务，对结构化的目标分析比较有效，比如图形

#关键代码：
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
