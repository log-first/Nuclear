import nucleus as nuc
import numpy as np
import pandas as pd

from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import tensorflow as tf

import matplotlib.pyplot as plt


def CNN_model(x_train, y_train, input_shape, epochs=10):
    # 搭建训练层
    # 卷积层、池化层
    # 没有提供更多的修改参数接口
    model = models.Sequential()  # 线性模型
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # 全连接层
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    # 
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(x_train, y_train, epochs=epochs)

    accuracy = history.history['accuracy']
    loss = history.history['loss']

    return model, accuracy, loss


