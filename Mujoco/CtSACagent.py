import tensorflow as tf
from keras import Model
from keras.layers import  Dense, Activation

class Actor_val(Model): 
    # 评估网络,输出动作
    def __init__(self,actions):
        super().__init__() 
        # resnet
        # self.ly_n=LayerNormalization()
        self.f1 = Dense(64, activation='relu',kernel_initializer='he_uniform')
        self.f4 = Dense(64, activation='relu',kernel_initializer='he_uniform')
        self.f2 = Dense(actions, activation=None,kernel_initializer='zeros') # 输出层
        # self.f2_2 = Dense(1, activation='relu',kernel_initializer='zeros') # 输出层
        self.var = tf.Variable(-0.5, trainable=True,dtype=tf.float32)
        # 加载网络
        self.checkpoint_save_path = "./disTD/model1/actor"
        # if os.path.exists(self.checkpoint_save_path + '.index'):
        #     print('-------------load the model-----------------')
        #     self.load_weights(self.checkpoint_save_path)
        # else:
        #     print('-------------train new model-----------------')
    @tf.function
    def call(self,x):
        x = self.f1(x)
        x1 = self.f4(x)
        y_mean = self.f2(x1) # 要考虑输入输出的维度 
        y_std =  self.var + tf.zeros_like(y_mean)  #- self.f2_2(x) +
        y_std = tf.math.exp(y_std)
        return y_mean,y_std

    def save_wei(self):
        # 保存网络
        self.save_weights(self.checkpoint_save_path)

class Critic_val(Model): 
    # 评估网络
    def __init__(self):
        super().__init__() 
        # resnet
        self.f1 = Dense(64, activation='relu',kernel_initializer='he_uniform')
        self.f4 = Dense(64, activation='relu',kernel_initializer='he_uniform')
        self.f2 = Dense(1, activation=None,kernel_initializer='zeros')
        # 加载网络
        self.checkpoint_save_path = "./disTD/model1/critic"
        # if os.path.exists(self.checkpoint_save_path + '.index'):
        #     print('-------------load the model-----------------')
        #     self.load_weights(self.checkpoint_save_path)
        # else:
        #     print('-------------train new model-----------------')
    @tf.function
    def call(self,s,a):
        x=tf.concat([s,a],1)
        x=self.f1(x)
        x=self.f4(x)
        y=self.f2(x) # 要考虑输入输出的维度
        return y        

    def save_wei(self):
        # 保存网络
        self.save_weights(self.checkpoint_save_path)


