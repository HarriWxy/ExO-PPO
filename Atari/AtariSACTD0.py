# import gymnasium as gym
import tensorflow as tf
from keras import losses
import keras.optimizers as optimizers
# import random
import numpy as np
from collections import deque
import os 
from AtariagentImEpSAC import  Actor_val,Critic_val#,ImagePro
import datetime
# import platform
# from ReplayBuffer import ReplayBuffer
import keras.mixed_precision  as mixed_precision
import envpool
import copy as cp

'modified disSAC. Base: Entropy -  pi * V * pi/beta  // V -> reward , clip + no kl'
'SAC algo.+TD0'


os.environ['CUDA_VISIBLE_DEVICES']='1'
ENV_NAME = 'Enduro-v5'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)])
GAMMA = 0.990 # 未来奖励的衰减
Replay_N = 8 # 观测存储器D的容量
BATCH = 128 # 训练batch大小
TAU=0.995
Run_Step = 64
Train_Env_num = 4
Test_Env_num = 10
Train_step = 8    # 8:4
OBSERVE = Train_Env_num * Replay_N -1  #* Run_Step // 256 训练前观察积累的轮数

class Memory:
    def __init__(self, obs_shape):
        self.size = Run_Step
        self.obses   = np.zeros((self.size+1, 4,84,84),dtype=np.float32)
        self.actions = np.zeros((self.size,1),dtype=np.int32)
        self.rewards = np.zeros((self.size),dtype=np.float32)
        self.dones   = np.zeros((self.size),dtype=np.int32)
        # self.values  = np.zeros((self.size),dtype=np.float32)
        self.policy  = np.zeros((self.size),dtype=np.float32)
        self.deltas  = np.zeros((self.size),dtype=np.float64)
        self.discounted_rew_sum = np.zeros((self.size),dtype=np.float32)
        # self.gae = np.zeros((self.size),dtype=np.float32)
        self.sample_i = 0  

    def __len__(self):
        return self.size

    def add(self, obs, action, reward, done, policy):
        if self.sample_i < self.size:
            self.rewards[self.sample_i] = reward
            self.dones[self.sample_i] = done
            self.policy[self.sample_i] = policy
            self.actions[self.sample_i] = action
        self.obses[self.sample_i] = obs
        self.sample_i += 1

    def compute_gae(self):  #, cri_net:Critic_val, act_net:Actor_val):
        self.obses_t = np.roll(self.obses,-1)[:self.size]
        # obs = tf.convert_to_tensor(self.obses,tf.float32)
        # values = cri_net(obs)
        # # values = tf.gather_nd(values,self.actions,batch_dims=1)
        # values = tf.squeeze(values) #- tf.reduce_mean(0.1* alpha * tf.math.log(act_net(obs) + 1e-6),1)
        # values = values.numpy()
        # values_t = np.roll(values,-1)[:self.size]
        # values = values[:self.size]
        # self.deltas = self.rewards + GAMMA * values_t * (1-self.dones) - values
        # self.gae[-1] = self.deltas[-1]
        # for t in reversed(range(self.size-1)):
        #     self.gae[t] = self.deltas[t] + (1 - self.dones[t]) * (GAMMA * 0.95) * self.gae[t + 1]
        # self.discounted_rew_sum = self.gae + values
        # self.gae = (self.gae - np.mean(self.gae)) / (np.std(self.gae) + 1e-8) 
        return

    def reset(self):
        self.sample_i = 0
        self.obses = np.zeros_like(self.obses)
        self.actions = np.zeros_like(self.actions)
        self.rewards = np.zeros_like(self.rewards)
        # self.values = np.zeros_like(self.values)
        self.policy = np.zeros_like(self.policy)
        self.deltas = np.zeros_like(self.deltas)
        self.discounted_rew_sum = np.zeros_like(self.discounted_rew_sum)
        # self.gae = np.zeros_like(self.gae)

def evaluate():
    s = eva_env.reset()[0].astype(dtype=np.float32)
    rew = np.zeros(Test_Env_num)
    rew_n = np.zeros(Test_Env_num)
    count = 0
    while count < Test_Env_num:
        xt = actor_val(s).numpy()
        action = tf.squeeze(tf.random.categorical(tf.math.log(xt),1,dtype=tf.int32),-1).numpy()
        s,r,d,ter,_= eva_env.step(action)
        # np.where(ter==True, r-10., r)
        d = np.logical_or(d,ter)
        rew += r 
        if np.any(d) :
            env_end = np.where(d)[0]
            s[env_end] = eva_env.reset(env_end)[0].astype(dtype=np.float32)
            for i in env_end:
                rew_n[count] = rew[i]
                count += 1
                if count == Test_Env_num:
                    break
            rew[env_end]=0
        s = s.astype(dtype=np.float32)
    return rew_n.mean(), rew_n.std()

def sample_inde():
    indices = np.random.permutation(Run_Step * Train_Env_num * Replay_N)
    for idx in range(0, Run_Step, BATCH):
        yield indices[idx:idx + BATCH]

def trainNet(istrain, isrender):
    # 创建网络
    # tf.random.set_seed(42)

    # 将每一轮的观测存在D中，之后训练从D中随机抽取batch个数据训练，以打破时间连续导致的相关性，保证神经网络训练所需的随机性。
    mem = [Memory(state_dim) for _ in range(Train_Env_num)]  # Memory
    D_mem = deque(maxlen = Replay_N * Train_Env_num)

    t=0
    # temp_t=0
    s=env.reset()[0].astype(dtype=np.float32)
    ep_reward = 0
    ep_best=0

    # tensorboard
    train_log_dir='logs/'+ENV_NAME+'/'+datetime.datetime.now().strftime("%m%d-%H%M%S")+'SAC'+str(Replay_N)
    train_sum_writer = tf.summary.create_file_writer(train_log_dir)

    eps=0
    while eps < 150000:
        for _ in range(Run_Step):
            xt = actor_val(s).numpy()
            action = tf.squeeze(tf.random.categorical(tf.math.log(xt),1,dtype=tf.int32),-1).numpy()
            s_t,r,d,ter,_= env.step(action)
            d = np.logical_or(d,ter)
            if np.any(d) :
                env_end = np.where(d)[0]
                s_t[env_end] = env.reset(env_end)[0].astype(dtype=np.float32)
            for i in range(Train_Env_num):
                mem[i].add(s[i],action[i],r[i],int(d[i]),xt[i][int(action[i])])
            s = s_t.astype(dtype=np.float32)
        t += Train_Env_num 
        for i in range(Train_Env_num):
            mem[i].add(s[i], None, None, None, None)
            # mem[i].compute_gae(critic_val1,actor_val)
            mem[i].compute_gae()
        D_mem.extend(cp.deepcopy(mem))
        
#============================ 训练网络 ===========================================
        # 观测一定轮数后开始训练
        if  t > OBSERVE and istrain : # 
            # 随机抽取minibatch个数据训练
            eps+=1
            obses = np.concatenate([i.obses[:-1] for i in D_mem],0)
            obses_t = np.concatenate([i.obses_t for i in D_mem],0)
            actions = np.concatenate([i.actions for i in D_mem],0)
            discounted_rew_sum = np.concatenate([i.discounted_rew_sum for i in D_mem],0)
            # policy = np.concatenate([i.policy for i in D_mem],0)
            dones = np.concatenate([i.dones for i in D_mem],0)
            for i in np.random.permutation(Train_step):
                # print("==================start train====================t=",t)
                # j = i % (Train_Env_num * Replay_N)
                for minibatch in sample_inde():
                    # 获得batch中的每一个变量
                    b_s = tf.convert_to_tensor(obses[minibatch])
                    b_a = tf.convert_to_tensor(actions[minibatch])
                    b_r = tf.convert_to_tensor(discounted_rew_sum[minibatch],dtype=tf.float32)
                    b_s_ = tf.convert_to_tensor(obses_t[minibatch])
                    b_done = tf.convert_to_tensor(dones[minibatch],dtype=tf.float32)
                    # b_ra = tf.convert_to_tensor(policy[minibatch],dtype=tf.float32)
                    # b_gae = tf.convert_to_tensor(gae[minibatch],dtype=tf.float32)
                    y = cal_y(b_s_,b_r,b_done)
                    loss1 = train_critic1(y,b_s,b_a)
                    loss2 = train_critic2(y,b_s,b_a)
                    # 更新actor
                    ac_loss = train_actor(b_s)
                    train_alpha(b_s)
                    alpha = tf.math.exp(log_alpha)
                # soft: ExponentialMovingAverage更新参数方法
                # if i == temp_t-1:
            if eps % 8 == 1:
                ep_reward, ep_std = evaluate()
                print("ep=",eps,"loss1 = %f" % loss1,"ac-loss = %f" % ac_loss,"alpha= %f" % alpha,"score=",ep_reward,'+',ep_std)

                    # tensorboard
                        
                with train_sum_writer.as_default():
                    tf.summary.scalar('std',ep_std,step = t)
                    tf.summary.scalar('score',ep_reward,step = t)            
            
            # t=0
            # 每1000轮保存一次网络参数
                if  ep_best < ep_reward or ep_reward > 200:
                    ep_best = ep_reward
                    print("=================model save====================")
        for i in range(Train_Env_num):
            mem[i].reset()

@tf.function
def cal_y(b_s_,b_r,b_done):
    a_temp = actor_val(b_s_)
    dq1 = tf.math.minimum(critic_tar1(b_s_),critic_tar2(b_s_)) - alpha * tf.math.log(a_temp+ 1e-6)
    tarQ = tf.reduce_sum(tf.multiply(a_temp,dq1),-1)
    y = b_r + GAMMA*tarQ* (tf.ones(BATCH,dtype=tf.float32) - b_done)
    return y

@tf.function
def train_critic1(y,b_s,b_a):
    # 训练Critic
    with tf.GradientTape() as tape:
        dq1 = tf.gather_nd(critic_val1(b_s),b_a,batch_dims=1)
        loss1 = losses.MAE(y,dq1) 
        loss = optimizer_cri1.get_scaled_loss(loss1)
    gradients = tape.gradient(loss, critic_val1.trainable_variables)
    gradients = optimizer_cri1.get_unscaled_gradients(gradients)
    optimizer_cri1.apply_gradients(zip(gradients, critic_val1.trainable_variables))
    return loss1

@tf.function
def train_critic2(y,b_s,b_a):
    with tf.GradientTape() as tape:
        dq1 = tf.gather_nd(critic_val2(b_s),b_a,batch_dims=1)
        loss2 = losses.MAE(y,dq1) 
        loss = optimizer_cri2.get_scaled_loss(loss2)
    gradients = tape.gradient(loss, critic_val2.trainable_variables)
    gradients = optimizer_cri2.get_unscaled_gradients(gradients)
    optimizer_cri2.apply_gradients(zip(gradients, critic_val2.trainable_variables))
    return loss2

@tf.function
def train_actor(b_s):
    with tf.GradientTape() as tape: 
        a_temp = actor_val(b_s)
        log_pis = tf.math.log(a_temp + 1e-8)
        dq1 = alpha*log_pis - tf.math.minimum(critic_val1(b_s),critic_val2(b_s))
        Q = tf.reduce_sum(tf.multiply(a_temp,dq1),-1)
        ac_loss = tf.reduce_mean(Q)
        ac_loss1 = optimizer_ac.get_scaled_loss(ac_loss)
    gradients = tape.gradient(ac_loss1, actor_val.trainable_variables)
    gradients = optimizer_ac.get_unscaled_gradients(gradients)
    optimizer_ac.apply_gradients(zip(gradients, actor_val.trainable_variables))
    dq1 = tf.reduce_mean(dq1)
    return dq1

@tf.function
def train_alpha(b_s):
    a_temp = actor_val(b_s)
    log_pis = tf.math.log(a_temp + 1e-8)
    with tf.GradientTape() as tape:
        loss = -tf.reduce_mean(log_alpha*(log_pis+target_entropy))
    gradients = tape.gradient(loss, [log_alpha])
    optimizer_alp.apply_gradients(zip(gradients, [log_alpha]))
    return loss

if __name__ == "__main__":
    render = True
    env =  envpool.make_gymnasium(ENV_NAME,episodic_life=True,reward_clip=True,num_envs=Train_Env_num)# gym.make(ENV_NAME)
    eva_env = envpool.make_gymnasium(ENV_NAME,num_envs=Test_Env_num)
    policy = mixed_precision.Policy('float32')
    mixed_precision.set_global_policy(policy)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    actor_val = Actor_val(action_dim)#, imnet)
    critic_val1 = Critic_val(action_dim)#imnet)
    critic_tar1 = Critic_val(action_dim)
    critic_tar1.set_weights(critic_val1.get_weights())
    critic_val2 = Critic_val(action_dim)
    critic_tar2 = Critic_val(action_dim)
    critic_tar2.set_weights(critic_val2.get_weights())
    # actor_tar = Actor_val(action_dim)#, imnet)
    log_alpha = tf.Variable(0.0)
    alpha = tf.math.exp(log_alpha)
    target_entropy = 0.4#-np.prod(action_dim)

    optimizer_alp = optimizers.Adam(learning_rate = 1e-4)
    lr_schedual = optimizers.schedules.ExponentialDecay(3e-4,8000,0.95,staircase=True)
    optimizer_cri1 = mixed_precision.LossScaleOptimizer(optimizers.Adam(learning_rate = lr_schedual,epsilon=1e-7))
    optimizer_cri2 = mixed_precision.LossScaleOptimizer(optimizers.Adam(learning_rate = 1e-5))
    optimizer_ac = mixed_precision.LossScaleOptimizer(optimizers.Adam(learning_rate = 1e-5))

    trainNet(True,render)
    # env=wrappers.Monitor(env,"./res2")
    # test()
    env.close()
