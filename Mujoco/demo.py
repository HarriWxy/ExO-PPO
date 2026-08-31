import gymnasium as gym
import tensorflow as tf
from keras import losses
import keras.optimizers as optimizers
import random
import numpy as np
from collections import deque
import os 
# from disSACagent import  Actor_val,Critic_val
import datetime
import time
# import platform
# from ReplayBuffer import ReplayBuffer
import keras.mixed_precision  as mixed_precision
'range 20 train'
'continious action space'

os.environ['CUDA_VISIBLE_DEVICES']='0'
ENV_NAME = 'Reacher-v4'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])

GAMMA = 0.990 # 未来奖励的衰减
 # 训练前观察积累的轮数
REPLAY_MEMORY = 1000000 # 观测存储器D的容量
BATCH = 256 # 训练batch大小
OBSERVE = BATCH + 500
TAU=0.995

# def evaluate():
#     s = eva_env.reset()[0]
#     rew = 0.
#     while True:
#         s = tf.expand_dims(tf.convert_to_tensor(s, dtype=tf.float32),0)
#         action = actor_val(s)[0]
#         action = int(np.random.choice(action_dim,1,p=action.numpy()))
#         # action = int(tf.argmax(action))
#         s,r,d,ter,_= eva_env.step(action)
#         if ter:
#             r = -10.
#         rew += r
#         if d or ter :
#             break
#     return rew

def trainNet(istrain, isrender):
    # 创建网络
    tf.random.set_seed(42)

    # 将每一轮的观测存在D中，之后训练从D中随机抽取batch个数据训练，以打破时间连续导致的相关性，保证神经网络训练所需的随机性。
    # mem = ReplayBuffer(REPLAY_MEMORY)  # Memory
    D = deque(maxlen=REPLAY_MEMORY)

    t=0
    temp_t=0
    s=env.reset(seed=42)[0]
    s=tf.convert_to_tensor(s, dtype=tf.float32)
    ep_reward = 0
    ep_best=0

    # tensorboard
    # train_log_dir='logs/ddpg/'+datetime.datetime.now().strftime("%m%d-%H-%M")
    # train_sum_writer = tf.summary.create_file_writer(train_log_dir)

    eps=0
    while eps < 150000:
        t+=1
        # 分段调整学习率
        # if eps==1000:
        #     optimizer_ac1 = tf.keras.optimizers.Adam(learning_rate = 1e-5)
        #     optimizer_ac2 = tf.keras.optimizers.Adam(learning_rate = 1e-5)
        #     optimizer_ac3 = tf.keras.optimizers.Adam(learning_rate = 1e-5)

        if isrender:
            # if platform.system() == 'Windows':
            #     pygame.event.get()
            env.render() # 显示
        
#         stemp = tf.expand_dims(s,0)
#         # xt = actor_val(stemp)[0]
#         'test'
#         xt = actor_val(stemp)
#         action = int(tf.random.categorical(tf.math.log(xt),1))
#         # alpha_t=np.exp2(-np.log10(np.log1p(t)))
#         # if  istrain:
#         #     action = int(np.random.choice(action_dim,1,p=xt.numpy()))
#         # else:
#         #     action = int(tf.argmax(xt))

#         s_t,r,d,ter,_= env.step(action)
#         s_t = tf.convert_to_tensor(s_t, dtype=tf.float32)
#         if ter:
#             r = -10
#         # if d and temp_t < 480:
#         #     r = -10
#         # ep_reward+=r
#         temp_t+=1
#         # print("score:",ep_reward)
#         # mem.add_transition((s,xt,r,s_t,d))
#         D.append((s,xt[0],r,s_t,d))
#         # if  temp_t > 2000: #
#         #     d=True
#         # if len(D) > REPLAY_MEMORY:
#         #     D.popleft()
#         s = s_t



# #============================ 训练网络 ===========================================
#         # 观测一定轮数后开始训练
#         if  t > OBSERVE and istrain and t % 20:#(d or ter):
#             # 随机抽取minibatch个数据训练
#             eps+=1
#             # random.shuffle(D)
#             ti = time.time()
#             for i in range(2000):#temp_t):
#                 # print("==================start train====================t=",t)

#                 # minibatch = mem.sample_minibatch(BATCH)
#                 minibatch = random.sample(D, BATCH)
#                 # 获得batch中的每一个变量
#                 b_s = tf.convert_to_tensor([d[0] for d in minibatch])
#                 b_a = tf.convert_to_tensor([d[1] for d in minibatch])
#                 b_r = tf.convert_to_tensor([d[2] for d in minibatch],dtype=tf.float32)
#                 # b_r = tf.cast(b_r,tf.float32)
#                 b_s_ = tf.convert_to_tensor([d[3] for d in minibatch])
#                 b_done = tf.convert_to_tensor([d[4] for d in minibatch],dtype=tf.float32)
#                 # b_done = tf.cast(b_done,tf.float32)
#                 y = cal_y(b_s_,b_r,b_done)
#                 loss1 = train_critic1(y,b_s,b_a)
#                 loss2 = train_critic2(y,b_s,b_a)
#                 # mem.update_weights(loss1)
#                 # 更新actor
#                 ac_loss = train_actor(b_s)
#                 train_alpha(b_s)
#                 alpha = tf.math.exp(log_alpha)
#                 # actor_tar.set_weights(actor_val.get_weights())
#                 if  i % 2 == 1:
#                 # soft: ExponentialMovingAverage更新参数方法
#                     cri_tar_var=[i*TAU+j*(1-TAU) for i, j in zip(critic_tar1.get_weights(),critic_val1.get_weights())]
#                     critic_tar1.set_weights(cri_tar_var)
#                     cri_tar_var=[i*TAU+j*(1-TAU) for i, j in zip(critic_tar2.get_weights(),critic_val2.get_weights())]
#                     critic_tar2.set_weights(cri_tar_var)
                    
#                 # if i == temp_t-1:
#             et = time.time()
#             print("time=",et-ti)
#             if eps % 10 == 0:
#                 ep_reward = evaluate()
#                 print("ep=",eps//10,"loss1 = %f" % loss1,"loss2 = %f" % loss2,"ac-loss = %f" % ac_loss,"alpha= %f" % alpha,"score=",ep_reward)

#                 # tensorboard
                    
#                     # with train_sum_writer.as_default():
#                     #     tf.summary.scalar('cri-loss1',loss1,step=eps)
#                     #     tf.summary.scalar('cri-loss2',loss2,step=eps)
#                     #     tf.summary.scalar('ac-loss',ac_loss,step=eps)
#                     #     tf.summary.scalar('score',ep_reward,step=eps)            


            
#             # 每1000轮保存一次网络参数
#                 if  ep_best < ep_reward or ep_reward > 200:
#                     ep_best=ep_reward
#                     print("=================model save====================")
#                 # actor_val.save_wei()
#                 # actor_tar.save_wei()
#                 # critic_val1.save_wei()
#                 # critic_val2.save_wei()


#         if d==True or ter==True:
#             s = env.reset()[0] # 完成后reset
#             s=tf.convert_to_tensor(s, dtype=tf.float32)
#             temp_t=0
#             ep_reward=0

# @tf.function
# def cal_y(b_s_,b_r,b_done):
#     a_temp = actor_val(b_s_)
#     dq1 = tf.math.minimum(critic_tar1(b_s_,a_temp),critic_tar2(b_s_,a_temp)) - alpha * tf.math.log(a_temp+ 1e-6)
#     tarQ = tf.reduce_sum(tf.multiply(a_temp,dq1),-1)
#     y = b_r + GAMMA*tarQ* (tf.ones(BATCH,dtype=tf.float32) - b_done)
#     return y

# @tf.function
# def train_critic1(y,b_s,b_a):
#     # 训练Critic
#     acts_dim = tf.expand_dims(tf.argmax(b_a,1),1)
#     # b_a = actor_val(b_s)+tf.clip_by_value(tf.random.normal([BATCH,action_dim],0,0.1,tf.float32), -0.5,0.5)
#     with tf.GradientTape() as tape:
#         dq1 = tf.gather_nd(critic_val1(b_s,b_a),acts_dim,batch_dims=1)
#         loss1 = losses.MAE(y,dq1) 
#         loss = optimizer_ac1.get_scaled_loss(loss1)
#         # print("loss1 = %f " % loss1)
#     gradients = tape.gradient(loss, critic_val1.trainable_variables)
#     gradients = optimizer_ac1.get_unscaled_gradients(gradients)
#     optimizer_ac1.apply_gradients(zip(gradients, critic_val1.trainable_variables))
#     return loss1

# @tf.function
# def train_critic2(y,b_s,b_a):
#     acts_dim = tf.expand_dims(tf.argmax(b_a,1),1)
#     # b_a = actor_val(b_s)+tf.clip_by_value(tf.random.normal([BATCH,action_dim],0,0.1,tf.float32), -0.5,0.5)
#     with tf.GradientTape() as tape:
#         dq1 = tf.gather_nd(critic_val2(b_s,b_a),acts_dim,batch_dims=1)
#         loss2 = losses.MAE(y,dq1) 
#         loss = optimizer_ac2.get_scaled_loss(loss2)
#         # print("loss2 = %f " % loss2)
#     gradients = tape.gradient(loss, critic_val2.trainable_variables)
#     gradients = optimizer_ac2.get_unscaled_gradients(gradients)
#     optimizer_ac2.apply_gradients(zip(gradients, critic_val2.trainable_variables))
#     return loss2

# @tf.function
# def train_actor(b_s):
#     with tf.GradientTape() as tape: #在这个空间里面计算梯度
#         a_temp = actor_val(b_s)
#         log_pis = tf.math.log(a_temp + 1e-6)
#         dq1 = alpha*log_pis - tf.math.minimum(critic_val1(b_s,a_temp),critic_val2(b_s,a_temp))
#         Q = tf.reduce_sum(tf.multiply(a_temp,dq1),-1)
#         ac_loss = tf.reduce_mean(Q)
#         ac_loss1 = optimizer_ac3.get_scaled_loss(ac_loss)
#         # print("ac-loss = %f" % ac_loss)
#     gradients = tape.gradient(ac_loss1, actor_val.trainable_variables)
#     gradients = optimizer_ac3.get_unscaled_gradients(gradients)
#     optimizer_ac3.apply_gradients(zip(gradients, actor_val.trainable_variables))
#     return ac_loss

# @tf.function
# def train_alpha(b_s):
#     a_temp = actor_val(b_s)
#     log_pis = tf.math.log(a_temp + 1e-6)
#     with tf.GradientTape() as tape:
#         loss = -tf.reduce_mean(log_alpha*(log_pis+target_entropy))
#     gradients = tape.gradient(loss, [log_alpha])
#     optimizer_alp.apply_gradients(zip(gradients, [log_alpha]))
#     return loss

if __name__ == "__main__":
    render = True
    env = gym.make(ENV_NAME,render_mode="human")
    # eva_env = gym.make(ENV_NAME)
    # policy = mix_policy.Policy('float32')
    # mix_policy.set_global_policy(policy)
    # # env.unwrapped() # 原始类可以不受作弊限制
    # # env.seed(random.randrange(10))
    # # state_dim = env.observation_space.shape[0]
    # action_dim = 4
    # # action_bound = env.action_space.high
    # # action_dim=4
    # actor_val=Actor_val(action_dim)
    # actor_tar=Actor_val(action_dim)
    # critic_val1=Critic_val(action_dim,1)
    # critic_tar1=Critic_val(action_dim,1,'tar')
    # critic_tar1.set_weights(critic_val1.get_weights())
    # critic_val2=Critic_val(action_dim,2)
    # critic_tar2=Critic_val(action_dim,2,'tar')
    # critic_tar2.set_weights(critic_val2.get_weights())
    # log_alpha = tf.Variable(-0.5)
    # alpha = tf.math.exp(log_alpha)
    # target_entropy = 0.4#-np.prod(action_dim)
    # # optimizer_ac1 = optimizers.Adam(learning_rate = 1e-5)
    # # optimizer_ac2 = optimizers.Adam(learning_rate = 1e-5)
    # optimizer_alp = optimizers.Adam(learning_rate = 2e-5)
    # optimizer_ac1 = mixed_precision.LossScaleOptimizer(optimizers.Adam(learning_rate = 1e-4))
    # optimizer_ac2 = mixed_precision.LossScaleOptimizer(optimizers.Adam(learning_rate = 1e-4))
    # optimizer_ac3 = mixed_precision.LossScaleOptimizer(optimizers.Adam(learning_rate = 1e-5))
    # optimizer_ac1.clipvalue=1.0
    # optimizer_ac2.clipvalue=1.0
    # optimizer_ac3.clipvalue=1.0

    trainNet(True,render)
    # env=wrappers.Monitor(env,"./res2")
    # test()
    env.close()

