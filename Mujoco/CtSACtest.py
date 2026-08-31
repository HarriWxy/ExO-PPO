import gym
import tensorflow as tf
import keras.optimizers as optimizers
import random
import numpy as np
from collections import deque
import os 
from Mujoco.CtSACagent import  Actor_val,Critic_val
from keras import losses
# from ReplayBuffer import ReplayBuffer
import keras.mixed_precision  as mixed_precision
# import tensorflow_probability.python.distributions as tfd

os.environ['CUDA_VISIBLE_DEVICES']='0'
ENV_NAME = 'LunarLanderContinuous-v2'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])

GAMMA = 0.990 # 未来奖励的衰减
OBSERVE = 1200 # 训练前观察积累的轮数
REPLAY_MEMORY = 1000000 # 观测存储器D的容量
BATCH = 128 # 训练batch大小
TAU=0.995

def trainNet(istrain, isrender):
    # 创建网络
    # tf.random.set_seed(42)

    # 将每一轮的观测存在D中，之后训练从D中随机抽取batch个数据训练，以打破时间连续导致的相关性，保证神经网络训练所需的随机性。
    # D = ReplayBuffer(REPLAY_MEMORY)  # Memory
    D = deque()

    t=0
    temp_t=0
    s=env.reset(seed=42)
    ep_reward = 0
    ep_best=0

    # tensorboard
    # train_log_dir='logs/ddpg/'+datetime.datetime.now().strftime("%m%d-%H-%M")+'res2'
    # train_sum_writer = tf.summary.create_file_writer(train_log_dir)

    eps=0
    while eps < 1500:
        # 分段调整学习率
        # if eps==1000:
            # optimizer_ac1 = tf.keras.optimizers.Adam(learning_rate = 1e-5)
            # optimizer_ac2 = tf.keras.optimizers.Adam(learning_rate = 1e-5)
            # optimizer_ac3 = tf.keras.optimizers.Adam(learning_rate = 1e-5)

        if isrender:
            # if platform.system() == 'Windows':
            #     pygame.event.get()
            env.render() # 显示
        
        stemp = tf.expand_dims(tf.convert_to_tensor(s, dtype=tf.float32),0)
        means, log_stds = actor_val(stemp) # 注意一下动作也有维度
        if istrain:
            std = tf.math.exp(log_stds)
            action = means + tf.random.normal(shape=means.shape, dtype=tf.float32) * std
            action = tf.nn.tanh(action)[0]
        else:
            action = means[0]

        s_t,r,d,_= env.step(action)
        ep_reward+=r
        temp_t+=1
        # print("score:",ep_reward)
        D.append((s,action,r,s_t,d))
        # if  temp_t > 2000: #
        #     d=True
        if len(D) > REPLAY_MEMORY:
            D.popleft()
        s = s_t
        t += 1


#============================ 训练网络 ===========================================
        # 观测一定轮数后开始训练
        if  t > OBSERVE and istrain:
            # 随机抽取minibatch个数据训练
            eps+=1
            random.shuffle(D)
            for i in range(temp_t):
                # print("==================start train====================t=",t)
                minibatch = random.sample(D, BATCH)
                # minibatch = D.sample_minibatch(BATCH)
                # 获得batch中的每一个变量
                b_s = tf.stack([d[0] for d in minibatch])
                b_a = tf.stack([d[1] for d in minibatch])
                b_r = tf.stack([d[2] for d in minibatch])
                b_r = tf.cast(b_r,tf.float32)
                b_s_ = tf.stack([d[3] for d in minibatch])
                b_done = tf.stack([d[4] for d in minibatch])
                b_done = tf.cast(b_done,tf.float32)
                y = cal_y(b_s_,b_r,b_done)
                loss1 = train_critic1(y,b_s,b_a)
                loss2 = train_critic2(y,b_s,b_a)
                # 更新actor
                # if  i % 2 == 1:
                ac_loss = train_actor(b_s)
                alpha_loss = train_alpha(b_s)
                alpha = tf.math.exp(log_alpha)
                # soft: ExponentialMovingAverage更新参数方法
                cri_tar_var=[i*TAU+j*(1-TAU) for i, j in zip(critic_tar1.get_weights(),critic_val1.get_weights())]
                critic_tar1.set_weights(cri_tar_var)
                cri_tar_var=[i*TAU+j*(1-TAU) for i, j in zip(critic_tar2.get_weights(),critic_val2.get_weights())]
                critic_tar2.set_weights(cri_tar_var)
                    
                if i == temp_t-1:
                    print("contin5,ep=",eps,"loss1 = %f" % loss1,"loss2 = %f" % loss2,"ac-loss = %f" % ac_loss,"alpha= %f" % alpha,"score=",ep_reward)
                
                # # tensorboard
                #     
                #     with train_sum_writer.as_default():
                #         tf.summary.scalar('cri-loss1',loss1,step=eps)
                #         tf.summary.scalar('cri-loss2',loss2,step=eps)
                #         tf.summary.scalar('ac-loss',ac_loss,step=eps)
                #         tf.summary.scalar('score',ep_reward,step=eps)            


            
            # 每1000轮保存一次网络参数
            if  ep_best < ep_reward or ep_reward > 200:
                ep_best=ep_reward
                print("=================model save====================")
                # actor_val.save_wei()
                # actor_tar.save_wei()
                # critic_val1.save_wei()
                # critic_val2.save_wei()
        if d==True:
            s = env.reset() # 完成后reset
            temp_t=0
            ep_reward=0

@tf.function
def process_actions(mean,log_std):
    std = tf.math.exp(log_std)
    raw_actions = mean + tf.random.normal(shape=mean.shape,dtype=tf.float32)*std
    # log_prob = tfd.Normal(loc=mean, scale=std).log_prob(raw_actions)
    log_prob = tf.math.log(tf.math.exp(-0.5*tf.square(raw_actions-mean)/tf.math.square(std))/tf.math.sqrt(2*np.pi*tf.math.square(std)))
    actions = tf.math.tanh(raw_actions)
    log_prob -= tf.math.log(1 - actions**2 + 1e-6)
    log_prob = tf.reduce_sum(log_prob,1)
    return actions,log_prob

@tf.function
def cal_y(b_s_,b_r,b_done): # compute Q
    acts,log_pis = actor_val(b_s_)# + tf.clip_by_value( tf.random.normal([BATCH,action_dim],0,0.1,tf.float32), -0.5,0.5)
    acts,log_pis = process_actions(acts,log_pis)
    valtemp = tf.reshape(tf.math.minimum(critic_tar1(b_s_,acts), critic_tar2(b_s_,acts)),[-1])-alpha*log_pis
    y = b_r + GAMMA *valtemp * (tf.ones(BATCH,dtype=tf.float32) - b_done)
    return y

@tf.function
def train_critic1(y,b_s,b_a):
    # 训练Critic
    with tf.GradientTape() as tape:
        loss1 = losses.MAE(y,tf.reshape(critic_val1(b_s,b_a),[-1])) 
        loss = optimizer_ac1.get_scaled_loss(loss1)
        # print("loss1 = %f " % loss1)
    gradients = tape.gradient(loss, critic_val1.trainable_variables)
    gradients = optimizer_ac1.get_unscaled_gradients(gradients)
    optimizer_ac1.apply_gradients(zip(gradients, critic_val1.trainable_variables))
    return loss1

@tf.function
def train_critic2(y,b_s,b_a):
    with tf.GradientTape() as tape:
        loss2 = losses.MAE(y,tf.reshape(critic_val2(b_s,b_a),[-1]))
        loss = optimizer_ac2.get_scaled_loss(loss2)
        # print("loss2 = %f " % loss2)
    gradients = tape.gradient(loss, critic_val2.trainable_variables)
    gradients = optimizer_ac2.get_unscaled_gradients(gradients)
    optimizer_ac2.apply_gradients(zip(gradients, critic_val2.trainable_variables))
    return loss2

@tf.function
def train_actor(b_s):
    with tf.GradientTape() as tape: #在这个空间里面计算梯度
        a_temp,log_pis = actor_val(b_s)
        a_temp,log_pis = process_actions(a_temp,log_pis)
        q = tf.reshape(tf.math.minimum(critic_val1(b_s,a_temp), critic_val2(b_s,a_temp)),[-1])
        ac_loss = tf.reduce_mean(alpha * log_pis - q)
        ac_loss1 = optimizer_ac3.get_scaled_loss(ac_loss)
        # print("ac-loss = %f" % ac_loss)
    gradients = tape.gradient(ac_loss1, actor_val.trainable_variables)
    gradients = optimizer_ac3.get_unscaled_gradients(gradients)
    optimizer_ac3.apply_gradients(zip(gradients, actor_val.trainable_variables))
    return ac_loss

@tf.function
def train_alpha(b_s):
    a_temp,log_pis = actor_val(b_s)
    _,log_pis = process_actions(a_temp,log_pis)
    with tf.GradientTape() as tape:
        loss = -tf.reduce_mean(log_alpha*(log_pis+target_entropy))
    gradients = tape.gradient(loss, [log_alpha])
    optimizer_alp.apply_gradients(zip(gradients, [log_alpha]))
    return loss

if __name__ == "__main__":
    render = False
    env=gym.make(ENV_NAME)
    policy = mixed_precision.Policy('float32')
    mixed_precision.set_global_policy(policy)
    # env.unwrapped() # 原始类可以不受作弊限制
    # env.seed(random.randrange(10))
    # state_dim = env.observation_space.shape[0]
    action_dim = 2 #env.action_space.shape[0]
    # action_bound = env.action_space.high
    # action_dim=4
    actor_val=Actor_val(action_dim)
    critic_val1=Critic_val(action_dim,1)
    critic_tar1=Critic_val(action_dim,1,'tar')
    critic_tar1.set_weights(critic_val1.get_weights())
    critic_val2=Critic_val(action_dim,2)
    critic_tar2=Critic_val(action_dim,2,'tar')
    critic_tar2.set_weights(critic_val2.get_weights())

    log_alpha = tf.Variable(0.0)
    alpha = tf.math.exp(log_alpha)
    target_entropy = -np.prod(action_dim)
    # optimizer_ac1 = optimizers.Adam(learning_rate = 1e-4)
    # optimizer_ac2 = optimizers.Adam(learning_rate = 1e-4)
    # optimizer_ac3 = optimizers.Adam(learning_rate = 1e-4)
    optimizer_alp = optimizers.Adam(learning_rate=5e-5)
    optimizer_ac1 = mixed_precision.LossScaleOptimizer(optimizers.Adam(learning_rate = 1e-4))
    optimizer_ac2 = mixed_precision.LossScaleOptimizer(optimizers.Adam(learning_rate = 1e-4))
    optimizer_ac3 = mixed_precision.LossScaleOptimizer(optimizers.Adam(learning_rate = 5e-5))

    trainNet(True,render)
    # env=wrappers.Monitor(env,"./res2")
    # test()
    env.close()

