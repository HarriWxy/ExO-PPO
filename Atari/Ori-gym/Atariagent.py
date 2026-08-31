import tensorflow as tf
from keras import Model
from keras.layers import  Dense, Activation,LayerNormalization,Conv2D,Flatten,AveragePooling2D,MaxPool2D,BatchNormalization
# import os 

class Actor_val(Model): 
    # 评估网络,输出动作
    def __init__(self,actions):
        super().__init__() 
        # resnet
        self.c_1_1 = Conv2D(filters=32, kernel_size=(3, 3),strides=2, padding='valid',activation='elu',input_shape=(210,160,3),
                            kernel_initializer='he_uniform')
        self.p_1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')  # 池化层
        self.l_1 = LayerNormalization(-1)
        self.c_2_1 = Conv2D(filters=32, kernel_size=(3, 3),strides=2, padding='valid',activation='elu',
                            kernel_initializer='he_uniform')
        self.c_2_2 = Conv2D(filters=32, kernel_size=(3, 3),strides=1, padding='same',kernel_initializer='he_uniform')
        self.a_2 = Activation('elu')
        self.l_2 = Conv2D(filters=64, kernel_size=1, padding='valid',strides=2,kernel_initializer='he_uniform',activation='elu')

        self.c_3_1 = Conv2D(filters=64, kernel_size=(3, 3),strides=1, padding='valid',activation='elu',
                            kernel_initializer='he_uniform')
        self.l_4 = Flatten()
        # self.f1 = Dense(256, activation='elu', kernel_initializer='he_uniform')
        self.f1_2 = Dense(64, activation='elu', kernel_initializer='he_uniform')
        self.f1_3 = Dense(actions, activation='softmax',kernel_initializer='he_uniform') # 输出层
        # 加载网络
        self.checkpoint_save_path = "./disTD/model1/actor"
        # if os.path.exists(self.checkpoint_save_path + '.index'):
        #     print('-------------load the model-----------------')
        #     self.load_weights(self.checkpoint_save_path)
        # else:
        #     print('-------------train new model-----------------')
    @tf.function
    def call(self,x):
        x = self.c_1_1(x)
        x = self.p_1(x)
        x = self.l_1(x)
        x1 = self.c_2_1(x)
        x = self.c_2_2(x1)
        x = self.a_2(x+x1)
        x = self.l_2(x)
        x = self.c_3_1(x)
        x = self.l_4(x)
        x = self.f1_2(x)
        y = self.f1_3(x)
        return y

    def save_wei(self):
        # 保存网络
        self.save_weights(self.checkpoint_save_path)

class Critic_val(Model): 
    # 评估网络
    def __init__(self):
        super().__init__() 
        self.c_1_1 = Conv2D(filters=32, kernel_size=(3, 3),strides=2, padding='valid',activation='elu',input_shape=(210,160,3),
                    kernel_initializer='he_uniform')
        self.p_1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')  # 池化层
        self.l_1 =LayerNormalization(-1)
        self.c_2_1 = Conv2D(filters=32, kernel_size=(3, 3),strides=2, padding='valid',activation='elu',
                            kernel_initializer='he_uniform')
        self.c_2_2 = Conv2D(filters=32, kernel_size=(3, 3),strides=1, padding='same',kernel_initializer='he_uniform')
        self.a_2 = Activation('elu')
        self.l_2 = Conv2D(filters=64, kernel_size=1, padding='valid',strides=2,kernel_initializer='he_uniform',activation='elu')

        self.c_3_1 = Conv2D(filters=64, kernel_size=(3, 3),strides=1, padding='valid',activation='elu',
                            kernel_initializer='he_uniform')
        # self.c_3_2 = Conv2D(filters=64, kernel_size=(3, 3),strides=1, padding='same',kernel_initializer='he_uniform')
        # self.a_3 = Activation('elu')
        # self.l_3 = Conv2D(filters=128, kernel_size=1, padding='valid',strides=2,kernel_initializer='he_uniform')
        
        # self.c_4_1 = Conv2D(filters=128, kernel_size=(3, 3),strides=2, padding='valid',activation='elu',
        #                     kernel_initializer='he_uniform')
        # self.c_4_2 = Conv2D(filters=128, kernel_size=(3, 3),strides=1, padding='same',kernel_initializer='he_uniform')
        # self.a_4 = Activation('elu')
        # self.p_4 = AveragePooling2D(pool_size=(3, 3), strides=2, padding='valid')  # 池化层
        self.l_4 = Flatten()

        # self.f1 = Dense(256, activation='elu', kernel_initializer='he_uniform')
        self.f1_2 = Dense(64, activation='elu', kernel_initializer='he_uniform')
        self.f1_3 = Dense(1, activation=None,kernel_initializer='he_uniform')
        # 加载网络
        # self.checkpoint_save_path = "./disTD/model1/critic"+str(k)+name
        # if os.path.exists(self.checkpoint_save_path + '.index'):
        #     print('-------------load the model-----------------')
        #     self.load_weights(self.checkpoint_save_path)
        # else:
        #     print('-------------train new model-----------------')
    @tf.function
    def call(self,x):
        x = self.c_1_1(x)
        x = self.p_1(x)
        x = self.l_1(x)
        x1 = self.c_2_1(x)
        x = self.c_2_2(x1)
        x = self.a_2(x+x1)
        x = self.l_2(x)

        x = self.c_3_1(x)
        # x = self.c_3_2(x1)
        # x = self.a_3(x+x1)
        # x = self.l_3(x)

        # x1 = self.c_4_1(x)
        # x = self.c_4_2(x1)
        # x = self.a_4(x+x1)
        # x = self.p_4(x)
        x = self.l_4(x)

        # x = self.f1(x)
        x = self.f1_2(x)
        y = self.f1_3(x)
        return y

    def save_wei(self):
        # 保存网络
        self.save_weights(self.checkpoint_save_path)


