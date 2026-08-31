import tensorflow as tf
from keras import Model
from keras.api.layers import  Dense, Activation,LayerNormalization,Conv2D,Flatten,AveragePooling2D,MaxPool2D,Layer
from vit_for_small_dataset import ViT
# import os 

class ImagePro(Layer):
    def __init__(self, inputshape):
        super().__init__() 
        # resnet
        self.c_1_1 = Conv2D(filters=32, kernel_size=3,strides=2, padding='same',activation='elu',
                            kernel_initializer='he_uniform',data_format="channels_first")
        # self.l_1 =LayerNormalization(0)
        self.p_1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same',data_format="channels_first")  # 池化层
        self.c_2_1 = Conv2D(filters=32, kernel_size=3,strides=2, padding='same',activation='elu',
                            kernel_initializer='he_uniform',data_format="channels_first")
        self.c_2_2 = Conv2D(filters=40, kernel_size=1,strides=1, padding='same',kernel_initializer='he_uniform',data_format="channels_first")
        # self.l_2 = Conv2D(filters=128, kernel_size=1, padding='valid',strides=2,kernel_initializer='he_uniform',activation='relu',data_format="channels_first")

        self.c_3_1 = Conv2D(filters=64, kernel_size=3,strides=2, padding='same',activation='elu',data_format="channels_first",
                            kernel_initializer='he_uniform')
        self.l_4 = Flatten()
        # self.a_2 = Activation('elu')


    # @tf.function
    def call(self,x):
        x = x / 255.
        x = self.c_1_1(x) # 42*42
        x1 = self.p_1(x)
        # x = self.l_1(x)
        x = self.c_2_1(x)
        x = tf.concat([x,x1],axis=1)
        # x = self.c_2_2(x1)
        # x = self.a_2(x+x1)
        # x = self.l_2(x)
        x = self.c_2_2(x)
        x = self.c_3_1(x)
        x = self.l_4(x)
        return x

class Actor_val(Model): 
    # 评估网络,输出动作
    def __init__(self, actions, inputshape): # , img_net:ViT
        super().__init__() 
        # resnet
        # self.f1 = Dense(256, activation='elu',kernel_initializer='he_uniform')
        # self.ima_net = ImagePro(inputshape)
        self.ima_net = ViT(image_size=inputshape[1], patch_size = 7, num_classes = 32, 
                           dim = 32, depth = 3, heads = 4, mlp_dim = 32, dim_head=32,
                           dropout = 0.0, emb_dropout = 0.0)# img_net
        # self.f4 = Dense(512, activation='relu',kernel_initializer='he_uniform')
        # self.f41 = Dense(32, activation='relu',kernel_initializer='he_uniform')
        # self.f31 = Dense(64, activation=None,kernel_initializer='he_uniform')
        self.f2 = Dense(actions, activation='softmax',kernel_initializer='zeros') # 输出层
        self.acti1 = Activation("elu")
        # 加载网络
        self.checkpoint_save_path = "./disTD/model1/actor"
        # if os.path.exists(self.checkpoint_save_path + '.index'):
        #     print('-------------load the model-----------------')
        #     self.load_weights(self.checkpoint_save_path)
        # else:
        #     print('-------------train new model-----------------')
    # @tf.function
    def call(self,x,training=True):
        x = x / 255.
        x = self.ima_net(x,training=training)
        # x = self.f4(x)
        # x1 = self.f31(x)
        # x1 = self.acti1(x+x1)
        # x1 = self.f41(x)
        x = self.acti1(x)
        y = self.f2(x)
        return y

    def save_wei(self):
        # 保存网络
        self.save_weights(self.checkpoint_save_path)

class Critic_val(Model): 
    # 评估网络
    def __init__(self, inputshape): # ,img_net:ViT
        super().__init__() 
        # self.f1 = Dense(256, activation='elu',kernel_initializer='he_uniform')
        # self.ima_net = ImagePro(inputshape)
        self.ima_net = ViT(image_size=inputshape[1], patch_size = 7,  num_classes = 32, 
                           dim = 32, depth = 3, heads = 4, mlp_dim = 32, dim_head=32,
                                        dropout = 0.0, emb_dropout = 0.0)
        # self.f31 = Dense(64, activation=None,kernel_initializer='he_uniform')
        # self.f4 = Dense(256, activation='relu',kernel_initializer='he_uniform')
        # self.f42 = Dense(32, activation='relu',kernel_initializer='he_uniform')
        self.f2 = Dense(1, activation = None, kernel_initializer='he_uniform')
        self.acti1 = Activation("elu")
        # 加载网络
        # self.checkpoint_save_path = "./disTD/model1/critic"
        # if os.path.exists(self.checkpoint_save_path + '.index'):
        #     print('-------------load the model-----------------')
        #     self.load_weights(self.checkpoint_save_path)
        # else:
        #     print('-------------train new model-----------------')
    # @tf.function
    def call(self,x,training=True):
        x = x / 255.
        x = self.ima_net(x,training=training)
        # x=self.f4(x)
        # x1=self.f31(x)
        # x1=self.acti1(x+x1)
        # x1=self.f42(x)
        # y=self.f2(x)
        x = self.acti1(x)
        y = self.f2(x)
        return y

    # def save_wei(self):
    #     # 保存网络
    #     self.save_weights(self.checkpoint_save_path)


