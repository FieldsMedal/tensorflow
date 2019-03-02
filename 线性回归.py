import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

plotdata = {"batchsize":[],"loss":[]}

#生成模拟数据

trainX = np.linspace(-1,1,100)
trainY = 2 * trainX + np.random.randn(100) * 0.3