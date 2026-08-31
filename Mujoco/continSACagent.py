import tensorflow as tf
import os
import numpy as np
from keras import Model
from keras.layers import  Dense, Activation,LayerNormalization

class Actor_val(Model): 
    # 评估网络,输出动作
    def __init__(self,actions,name='val'):
        super().__init__() 
        # resnet
        # self.ly_n=LayerNormalization()
        self.f1 = Dense(64, activation='elu',kernel_initializer='he_uniform')
        self.f4 = Dense(16, activation='elu',kernel_initializer='he_uniform')
        self.f41 = Dense(4, activation='elu',kernel_initializer='he_uniform')
        # self.f3 = Dense(256, activation=None,
        #                    kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=42),
        #                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=42))
        self.f31 = Dense(16, activation=None,kernel_initializer='he_uniform')
        self.f2 = Dense(actions, activation=None,kernel_initializer='he_uniform') # 输出层
        self.f2_2 = Dense(actions, activation=None,kernel_initializer='he_uniform') # 输出层
        self.acti1 = Activation("elu")
        # 加载网络
        self.checkpoint_save_path = "./disTD/model1/actor"+name
        # if os.path.exists(self.checkpoint_save_path + '.index'):
        #     print('-------------load the model-----------------')
        #     self.load_weights(self.checkpoint_save_path)
        # else:
        #     print('-------------train new model-----------------')
    @tf.function
    def call(self,x):
        x=self.f1(x)
        # x=self.ly_n(x)
        # x1=self.f1(x)
        x = self.f4(x)
        x1 = self.f31(x)
        x1 = self.acti1(x+x1)
        x1 = self.f41(x1)
        y_mean = self.f2(x1) # 要考虑输入输出的维度
        y_logstd = self.f2_2(x1)
        return y_mean,y_logstd

    def save_wei(self):
        # 保存网络
        self.save_weights(self.checkpoint_save_path)
    def save_model(self):
        self.save(self.checkpoint_save_path+'.h5')

class Critic_val(Model): 
    # 评估网络
    def __init__(self,actions,k=1,name='val'):
        super().__init__() 
        # resnet
        # self.ly_n1=LayerNormalization()
        self.f1 = Dense(64, activation='elu',kernel_initializer='he_uniform')
        # self.f3 = Dense(256, activation=None,
        #                    kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=k+41),
        #                    bias_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=k+41))
        self.f31 = Dense(16, activation=None,kernel_initializer='he_uniform')
        self.f4 = Dense(16, activation='elu',kernel_initializer='he_uniform')
        self.f42 = Dense(4, activation='elu',kernel_initializer='he_uniform')
        self.f2 = Dense(1, activation=None,kernel_initializer='he_uniform')
        self.acti1 = Activation("elu")
        # 加载网络
        self.checkpoint_save_path = "./disTD/model1/critic"+str(k)+name
        # if os.path.exists(self.checkpoint_save_path + '.index'):
        #     print('-------------load the model-----------------')
        #     self.load_weights(self.checkpoint_save_path)
        # else:
        #     print('-------------train new model-----------------')
    @tf.function
    def call(self,s,a):
        x=tf.concat([s,a],1)
        # x=self.f3(x)
        x=self.f1(x)
        # x=self.ly_n1(x)
        x=self.f4(x)
        x1=self.f31(x)
        x1=self.acti1(x+x1)
        x1=self.f42(x1)
        y=self.f2(x1) # 要考虑输入输出的维度
        return y        

    def save_wei(self):
        # 保存网络
        self.save_weights(self.checkpoint_save_path)


