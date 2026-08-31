import tensorflow as tf
from keras import Model
from keras.api.layers import  Dense
import keras
# import os 


class Actor_val(Model): 
    # 评估网络,输出动作
    def __init__(self, actions):
        super().__init__() 
        # resnet
        # self.f1 = Dense(256, activation='elu',kernel_initializer='he_uniform')
        # self.f4 = Dense(256, activation='relu',kernel_initializer='he_uniform')
        self.f31 = Dense(64, activation='relu',kernel_initializer='he_uniform')
        self.f41 = Dense(64, activation='relu',kernel_initializer='he_uniform')
        self.f2_m = Dense(actions, activation=None,kernel_initializer='zeros') # 输出层
        # self.f2_v = Dense(1, activation='relu',kernel_initializer='zeros')
        # self.f2_v = Dense(actions, activation=None,kernel_initializer='zeros')
        initializer = keras.initializers.Constant(value=-1.2)
        self.var = self.add_weight(name='var', shape=(actions,), initializer=initializer, trainable=True)
        # self.var = -1.5 * self.var

        # tf.tf.Variable([-1.05]*actions, trainable=True, dtype=tf.float32)
        # self.acti1 = Activation("elu")
        # 加载网络
        self.checkpoint_save_path = "./disTD/model1/actor"
        # if os.path.exists(self.checkpoint_save_path + '.index'):
        #     print('-------------load the model-----------------')
        #     self.load_weights(self.checkpoint_save_path)
        # else:
        #     print('-------------train new model-----------------')
    # @tf.function
    def call(self,x:tf.Tensor):
        x = self.f31(x)
        x = self.f41(x)
        mean = self.f2_m(x)
        # log_var = self.f2_v(x)
        var = tf.exp(self.var)
        var = tf.clip_by_value(var,0.1,1) + tf.zeros_like(mean,dtype=tf.float32)
        # x1 = self.f41(x)+ self.f2_v(x) #
        # mean = self.f2_m(x1)
        # # mean = tf.clip_by_value(mean,-1,1)
        # # var = tf.clip_by_value(self.f2_v(x1),-0.9,1)
        # # var = 0.5*+0.3
        # var = tf.math.exp(self.var + tf.zeros_like(mean,dtype=tf.float32)) 
        # return mean,var
        return mean, var

    def save_wei(self):
        # 保存网络
        self.save_weights(self.checkpoint_save_path)

class Critic_val(Model): 
    # 评估网络
    def __init__(self):
        super().__init__() 
        # self.f1 = Dense(256, activation='elu',kernel_initializer='he_uniform')
        self.f31 = Dense(64, activation='relu',kernel_initializer='he_uniform')
        # self.f4 = Dense(256, activation='relu',kernel_initializer='he_uniform')
        self.f42 = Dense(64, activation='relu',kernel_initializer='he_uniform')
        self.f2 = Dense(1, activation=None,kernel_initializer='zeros')
        # self.acti1 = Activation("elu")
        # 加载网络
        # self.checkpoint_save_path = "./disTD/model1/critic"
        # if os.path.exists(self.checkpoint_save_path + '.index'):
        #     print('-------------load the model-----------------')
        #     self.load_weights(self.checkpoint_save_path)
        # else:
        #     print('-------------train new model-----------------')
    # @tf.function
    def call(self,x):
        # x = self.f1(x)
        x = self.f31(x)
        # x1=self.acti1(x+x1)
        x1 = self.f42(x)
        y = self.f2(x1)
        return y

    # def save_wei(self):
    #     # 保存网络
    #     self.save_weights(self.checkpoint_save_path)


