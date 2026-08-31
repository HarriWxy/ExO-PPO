# import gymnasium as gym
import tensorflow as tf
from keras import losses,optimizers
# import keras.optimizers as optimizers
import random
import numpy as np
from collections import deque
import os 
from AtariagentImEp import  Actor_val,Critic_val
import datetime
# import platform
# from ReplayBuffer import ReplayBuffer
# import keras.mixed_precision  as mixed_precision
import envpool
import copy as cp
from numba import njit

' Base: Entropy -  pi * V * pi/beta  // V -> reward ,no clip + kl (instant)'
'deque for off-memory'

'my algo.'

os.environ['CUDA_VISIBLE_DEVICES']='0'
ENV_NAME = 'Breakout-v5'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
GAMMA = 0.990 # 未来奖励的衰减
Replay_N = 4 # 观测存储器D的容量
BATCH = 256 # 训练batch大小
Lam_Gae = 0.99
Run_Step = 128
Train_Env_num = 8 * 2// Replay_N 
Test_Env_num = 4
EPS = 24800 * 2 // Train_Env_num 
Rec_Num = 32 * 8 * 4 // Train_Env_num #
Train_step = 1  # 8:4
OBSERVE = Train_Env_num * Replay_N * Run_Step -1  #* Run_Step // 256 训练前观察积累的轮数
beta = 5
klra = 1
Rec_step = Train_Env_num * Run_Step * Rec_Num
All_step = Train_Env_num * Run_Step * EPS

class Memory:
    def __init__(self, obs_shape):
        self.size = Run_Step
        self.obses   = np.zeros((self.size+1, state_dim[0], state_dim[1], state_dim[2]),dtype=np.float32)
        self.actions = np.zeros((self.size,1),dtype=np.int32)
        self.rewards = np.zeros((self.size),dtype=np.float32)
        self.dones   = np.zeros((self.size),dtype=np.int32)
        # self.values  = np.zeros((self.size),dtype=np.float32)
        self.policy  = np.zeros((self.size,action_dim),dtype=np.float32)
        self.deltas  = np.zeros((self.size),dtype=np.float64)
        self.discounted_rew_sum = np.zeros((self.size),dtype=np.float32)
        self.gae = np.zeros((self.size),dtype=np.float32)
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

    def compute_gae(self, net:Critic_val):
        values = net(tf.convert_to_tensor(self.obses,tf.float32))
        # values = tf.gather_nd(values,self.actions,batch_dims=1)
        values = tf.squeeze(values).numpy()
        values_t = np.roll(values,-1)[:self.size]
        values = values[:self.size]
        self.deltas = self.rewards + GAMMA * values_t * (1-self.dones) - values
        self.gae[-1] = self.deltas[-1]
        for t in reversed(range(self.size-1)):
            self.gae[t] = self.deltas[t] + (1 - self.dones[t]) * (GAMMA * Lam_Gae) * self.gae[t + 1]
        self.discounted_rew_sum = self.gae + values
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
        self.gae = np.zeros_like(self.gae)

def evaluate():
    s = eva_env.reset()[0].astype(dtype=np.float32)
    rew = np.zeros(Test_Env_num)
    rew_n = np.zeros(Test_Env_num)
    count = 0
    done_envs = []
    while count<Test_Env_num:
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
                if i in done_envs:
                    continue
                done_envs.append(i)
                rew_n[count] = rew[i]
                count += 1
                if count == Test_Env_num:
                    break
            rew[env_end] = 0
        s = s.astype(dtype=np.float32)
    # print(done_envs)
    return rew_n.mean(), rew_n.std()

@njit
def sample_inde():
    indices = np.random.permutation(Run_Step * Train_Env_num * Replay_N)
    # a = Run_Step * 8
    # b = min(indices.size, a)
    for idx in range(0, indices.size, BATCH): # b
        yield indices[idx:idx + BATCH]

def trainNet(istrain):
    # 创建网络
    # tf.random.set_seed(42)

    # 将每一轮的观测存在D中，之后训练从D中随机抽取batch个数据训练，以打破时间连续导致的相关性，保证神经网络训练所需的随机性。
    mem = [Memory(state_dim) for _ in range(Train_Env_num)]  # Memory
    D_mem:deque[Memory] = deque(maxlen = Replay_N * Train_Env_num)

    t = 0
    # temp_t=0
    s = env.reset()[0].astype(dtype=np.float32)
    ep_reward = 0
    ep_best = 0

    # tensorboard
    train_log_dir='logs/'+ENV_NAME+'/'+datetime.datetime.now().strftime("%m%d-%H%M%S")+'-beta:'+str(beta)+'-re:'+str(Replay_N)+'-entrodfsafdsae'+str(seed)
    train_sum_writer = tf.summary.create_file_writer(train_log_dir)

    eps = 0
    reward = np.zeros(Train_Env_num,np.float32)
    while eps < EPS:
        for _ in range(Run_Step):
            xt = actor_val(s)#.numpy()
            action = tf.squeeze(tf.random.categorical(tf.math.log(xt),1,dtype=tf.int32),-1).numpy()
            s_t,r,d,ter,_= env.step(action)
            # np.where(ter==True,r-10.,r)
            reward += r
            d = np.logical_or(d,ter)
            if np.any(d) :
                env_end = np.where(d)[0]
                s_t[env_end] = env.reset(env_end)[0].astype(dtype=np.float32)
            for i in range(Train_Env_num):
                mem[i].add(s[i],action[i],r[i],int(d[i]),xt[i])
            s = s_t.astype(dtype=np.float32)
        t += Train_Env_num * Run_Step
        for i in range(Train_Env_num):
            mem[i].add(s[i], None, None, None, None)
            mem[i].compute_gae(critic_val)
        D_mem.extend(cp.deepcopy(mem))
        
#============================ 训练网络 ===========================================
        # 观测一定轮数后开始训练
        if  t > OBSERVE and istrain : # 
            # 随机抽取minibatch个数据训练
            eps += 1
            obses = np.concatenate([i.obses[:-1] for i in D_mem],0)
            actions = np.concatenate([i.actions for i in D_mem],0)
            discounted_rew_sum = np.concatenate([i.discounted_rew_sum for i in D_mem],0)
            policy = np.concatenate([i.policy for i in D_mem],0)
            gae = np.concatenate([i.gae for i in D_mem],0)
            for i in range(Train_step):
                # print("==================start train====================t=",t)
                # j = i % (Train_Env_num * Replay_N)
                # mem_temp:Memory = D_mem[j]
                # mem_temp.compute_gae(critic_val1)
                for minibatch in sample_inde():
                    # 获得batch中的每一个变量
                    b_s = tf.convert_to_tensor(obses[minibatch])
                    b_a = tf.convert_to_tensor(actions[minibatch])
                    b_r = tf.convert_to_tensor(discounted_rew_sum[minibatch],dtype=tf.float32)
                    # b_s_ = tf.convert_to_tensor([d[3] for d in minibatch])
                    # b_done = tf.convert_to_tensor([d[3] for d in minibatch],dtype=tf.float32)
                    b_ra = tf.convert_to_tensor(policy[minibatch],dtype=tf.float32)
                    b_gae = gae[minibatch]
                    b_gae = (b_gae - b_gae.mean())/(b_gae.std()+1e-8)
                    b_gae = tf.convert_to_tensor(b_gae,dtype=tf.float32)
                    # y = cal_y(b_s_,b_r,b_done)
                    # if j > Train_Env_num * (Replay_N - 1) - 1:
                    #     for _ in range(Replay_N):
                    loss1 = train_critic(b_r,b_s)
                    # loss2 = train_critic2(y,b_s)
                    # 更新actor
                    # if  i % 2 == 1:
                    ac_loss = train_actor(b_s,b_a,b_ra,b_gae)
                    # train_alpha(b_s)
                    # alpha = 9*tf.math.exp(log_alpha)+1
                # soft: ExponentialMovingAverage更新参数方法
                # act_tar_var=[i*TAU+j*(1-TAU) for i, j in zip(actor_tar.get_weights(),actor_val.get_weights())]
                # train_kl(b_s)
                # if i % 4 == 0:
            # actor_tar.set_weights(actor_val.get_weights()) # act_tar_var) 
                # if i == temp_t-1:
            if eps % Rec_Num == 1: # 一个eps 512
                ep_reward, ep_std = evaluate()
                aver_rew = reward.mean() / Rec_Num
                reward = np.zeros_like(reward)
                print("ep=",eps,"loss1 = %f" % loss1,"ac-loss = %f" % ac_loss,"score=",ep_reward,'+',ep_std,'aver_rew=',aver_rew)

                    # tensorboard
                        
                with train_sum_writer.as_default():
                    tf.summary.scalar('rate',ac_loss,step = t)
                    tf.summary.scalar('aver_rew',aver_rew,step = t)
                    tf.summary.scalar('score',ep_reward,step = t)            
            
            # t=0
            # 每1000轮保存一次网络参数
                if  ep_best < ep_reward : #or ep_reward > 200
                    ep_best = ep_reward
                    print("=================model save====================")
        for i in range(Train_Env_num):
            mem[i].reset()

@tf.function
def train_critic(y,b_s):
    # 训练Critic
    # acts_dim = tf.expand_dims(tf.argmax(b_a,1),1)
    # b_a = actor_val(b_s)+tf.clip_by_value(tf.random.normal([BATCH,action_dim],0,0.1,tf.float32), -0.5,0.5)
    with tf.GradientTape() as tape:
        dq1 = tf.squeeze(critic_val(b_s))
        loss = losses.mean_squared_error(y,dq1) 
        # loss = optimizer_cri.get_scaled_loss(loss1)
        # print("loss1 = %f " % loss1)
    gradients = tape.gradient(loss, critic_val.trainable_variables)
    # gradients = optimizer_cri.get_unscaled_gradients(gradients)
    optimizer_cri.apply(gradients, critic_val.trainable_variables)
    return loss

@tf.function
def train_actor(b_s,b_a,b_ra,b_gae):
    with tf.GradientTape() as tape: 
        a_temp = actor_val(b_s)
        log_pis = tf.math.log(a_temp + 1e-8)
        dq1 = tf.divide(tf.gather_nd(a_temp,b_a,batch_dims=1),tf.gather_nd(b_ra,b_a,batch_dims=1) + 1e-8) # rate = pi/beta
        # dq2 = tf.where(dq1 >  0.2 + dq0,  0.2 + dq0 + 1/beta - tf.exp(beta * (0.2+dq0-dq1))/beta, dq1) # 需要先算大的(-dq1)的部分, 避免exp数值太大
        # dq2 = tf.where(dq2 < -0.2 + dq0, -0.2 + dq0 - 1/beta + tf.exp(beta * (dq2-(dq0-0.2)))/beta, dq2)
        # Q = - tf.minimum(tf.multiply(dq1, b_gae),tf.multiply(dq2,b_gae))
        dq2 = tf.where(dq1 > 1.2, 1.2 + 1/beta - tf.exp(beta * (1.2-dq1))/beta, dq1)
        dq2 = tf.where(dq2 < 0.8, 0.8 - 1/beta + tf.exp(beta * (dq2-0.8))/beta, dq2)
        # Q = tf.reduce_sum(tf.multiply(a_temp,alpha*log_pis),-1) - tf.multiply(dq2,dq1)
        dq2 = tf.minimum(tf.multiply(dq1, b_gae),tf.multiply(dq2,b_gae)) # tf.multiply(dq1, b_gae)clip  # tf.minimum(tf.multiply(dq2, b_gae),clip_adv)
        Q = tf.reduce_sum(tf.multiply(a_temp,log_pis),-1) # Entropy 
        Q = 0.01*Q - dq2 + klra * losses.kl_divergence(b_ra,a_temp) # pi * (Entropy -Q) * rate
        ac_loss = tf.reduce_mean(Q)
        # ac_loss1 = optimizer_ac.get_scaled_loss(ac_loss)
        # print("ac-loss = %f" % ac_loss)
    gradients = tape.gradient(ac_loss, actor_val.trainable_variables)
    # gradients = optimizer_ac.get_unscaled_gradients(gradients)
    optimizer_ac.apply(gradients, actor_val.trainable_variables)
    dq1 = tf.reduce_mean(tf.math.abs(tf.math.log(dq1))) #dq1) - tf.reduce_min(dq1)
    return dq1


if __name__ == "__main__":
    seed = 0    # 0 42 1024 
    # for seed in seeds:
    env =  envpool.make_gymnasium(ENV_NAME,num_envs=Train_Env_num,episodic_life=True,reward_clip=True)# gym.make(ENV_NAME)
    eva_env = envpool.make_gymnasium(ENV_NAME,num_envs=Test_Env_num,reward_clip=False) # 
    # policy = mixed_precision.Policy('float32')
    # policy = mixed_precision.set_global_policy(policy)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n

    actor_val = Actor_val(action_dim,state_dim)#, imnet)
    critic_val = Critic_val(state_dim)#imnet)
    # mixed_precision.LossScaleOptimizer(mixed_precision.LossScaleOptimizer(
    lr_schedual = optimizers.schedules.ExponentialDecay(2.5e-4,5000,0.98,staircase=True)
    optimizer_ac = optimizers.Adam(learning_rate = lr_schedual,epsilon=1e-8)
    lr_schedual_ = optimizers.schedules.ExponentialDecay(2e-4,5000,0.98,staircase=True)
    optimizer_cri = optimizers.Adam(learning_rate = lr_schedual_,epsilon=1e-8)
    trainNet(True)
        
