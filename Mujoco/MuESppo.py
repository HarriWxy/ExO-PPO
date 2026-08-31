import gymnasium as gym
import tensorflow as tf
from keras import losses
import keras.optimizers as optimizers
import random
import numpy as np
import os 
from Mujoco.MuagentImEp import  Actor_val,Critic_val
import datetime
import keras.mixed_precision  as mixed_precision
import envpool

'EnvPool'
'modified disSAC. Base: Entropy -  pi * V * clip(pi/beta)  // V -> reward'
'value clip, adv clip, adv norm, P3O + kl'

os.environ['CUDA_VISIBLE_DEVICES']='0'
ENV_NAME = 'BeamRider-v5'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])

GAMMA = 0.990 # 未来奖励的衰减
# REPLAY_MEMORY = 8192 # 观测存储器D的容量
BATCH = 128 # 训练batch大小
TAU=0.995
Run_Step = 256
Train_Env_num = 8
Test_Env_num = 5
Train_step = 4 #* Train_Env_num

class Memory:
    def __init__(self, obs_shape):
        self.size = Run_Step
        self.obses   = np.zeros((self.size+1, 4,84,84),dtype=np.float32)
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
            self.gae[t] = self.deltas[t] + (1 - self.dones[t]) * (GAMMA * 0.95) * self.gae[t + 1]
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

def sample_inde():
    indices = np.random.permutation(Run_Step * Train_Env_num)
    for idx in range(0, indices.size, BATCH):
        yield indices[idx:idx + BATCH]

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

def trainNet(istrain, isrender):
    # 将每一轮的观测存在D中，之后训练从D中随机抽取batch个数据训练，以打破时间连续导致的相关性，保证神经网络训练所需的随机性。
    mem = [Memory(state_dim) for _ in range(Train_Env_num)]  # Memory

    t = 0
    s = env.reset()[0].astype(dtype=np.float32)
    ep_reward = 0
    ep_best=0

    # tensorboard
    train_log_dir='logs/'+ENV_NAME+'/'+datetime.datetime.now().strftime("%m%d-%H%M%S")+'ESPPO-kl'
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
                mem[i].add(s[i],action[i],r[i],int(d[i]),xt[i])
            s = s_t.astype(dtype=np.float32)
        t += Train_Env_num * Run_Step
        for i in range(Train_Env_num):
            mem[i].add(s[i], None, None, None, None)
            mem[i].compute_gae(critic_val1)
        # for i in range(Run_Step):
        #     D_mem.append((mem.obses[i],[mem.actions[i]],mem.discounted_rew_sum[i],mem.dones[i],mem.policy[i],mem.gae[i]))

#============================ 训练网络 ===========================================
        # 观测一定轮数后开始训练
        if istrain :
            # 随机抽取minibatch个数据训练
            eps+=1
            obses = np.concatenate([i.obses[:-1] for i in mem],0)
            actions = np.concatenate([i.actions for i in mem],0)
            discounted_rew_sum = np.concatenate([i.discounted_rew_sum for i in mem],0)
            policy = np.concatenate([i.policy for i in mem],0)
            gae = np.concatenate([i.gae for i in mem],0)
            for i in range(Train_step):
                # j = i % Train_Env_num
                for minibatch in sample_inde():

                # 获得batch中的每一个变量
                    b_s = tf.convert_to_tensor(obses[minibatch])
                    b_a = tf.convert_to_tensor(actions[minibatch])
                    b_r = tf.convert_to_tensor(discounted_rew_sum[minibatch],dtype=tf.float32)
                    b_ra = tf.convert_to_tensor(policy[minibatch],dtype=tf.float32)
                    # b_gae = tf.convert_to_tensor(gae[minibatch],dtype=tf.float32)
                    b_gae = gae[minibatch]
                    b_gae = (b_gae - b_gae.mean())/(b_gae.std()+1e-8)
                    b_gae = tf.convert_to_tensor(b_gae,dtype=tf.float32)
                    loss1 = train_critic(b_r,b_s)
                    # 更新actor
                    ac_loss = train_actor(b_s,b_a,b_ra,b_gae)
                    if ac_loss > 0.25:
                        break

            if eps % 32==1:  # 一个 eps = 1024 step
                ep_reward, ep_std = evaluate()
                print("ep=",eps,"loss1 = %f" % loss1,"ac-loss = %f" % ac_loss,"score=",ep_reward,'+',ep_std)

                # tensorboard
                with train_sum_writer.as_default():
                    tf.summary.scalar('rate',ac_loss,step = t)
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
def train_critic(y,b_s):
    # 训练Critic
    # acts_dim = tf.expand_dims(tf.argmax(b_a,1),1)
    # b_a = actor_val(b_s)+tf.clip_by_value(tf.random.normal([BATCH,action_dim],0,0.1,tf.float32), -0.5,0.5)
    with tf.GradientTape() as tape:
        dq1 = tf.squeeze(critic_val1(b_s))
        loss1 = losses.MAE(y,dq1) 
        loss = optimizer_cri.get_scaled_loss(loss1)
        # print("loss1 = %f " % loss1)
    gradients = tape.gradient(loss, critic_val1.trainable_variables)
    gradients = optimizer_cri.get_unscaled_gradients(gradients)
    optimizer_cri.apply_gradients(zip(gradients, critic_val1.trainable_variables))
    return loss1

@tf.function
def train_actor(b_s,b_a,b_ra,b_gae):
    'return the rate'
    with tf.GradientTape() as tape: #在这个空间里面计算梯度
        a_temp = actor_val(b_s)
        log_pis = tf.math.log(a_temp + 1e-8)
        dq1 = tf.divide(tf.gather_nd(a_temp,b_a,batch_dims=1), tf.gather_nd(b_ra,b_a,batch_dims=1) + 1e-8) # rate = pi/beta
        Entr = tf.reduce_sum(tf.multiply(a_temp, alpha*log_pis),-1) # Entropy 
        dq2 = tf.multiply(dq1, b_gae)
        Q = 0.01*Entr - dq2 + losses.kl_divergence(b_ra,a_temp) # pi * (Entropy -Q) * rate 
        ac_loss = tf.reduce_mean(Q)
        ac_loss1 = optimizer_ac.get_scaled_loss(ac_loss)
        # print("ac-loss = %f" % ac_loss)
    gradients = tape.gradient(ac_loss1, actor_val.trainable_variables)
    gradients = optimizer_ac.get_unscaled_gradients(gradients)
    optimizer_ac.apply_gradients(zip(gradients, actor_val.trainable_variables))
    dq1 = tf.reduce_mean(tf.math.abs(tf.math.log(dq1)))
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
    render = False
    env =  envpool.make_gymnasium(ENV_NAME, episodic_life=True,reward_clip=True,num_envs=Train_Env_num,seed=4213)# gym.make(ENV_NAME)
    eva_env = envpool.make_gymnasium(ENV_NAME,num_envs=Test_Env_num,seed=4213)
    policy = mixed_precision.Policy('float32')
    mixed_precision.set_global_policy(policy)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    # action_bound = 
    # action_dim=4
    # imnet = ImagePro()
    actor_val = Actor_val(action_dim) #, imnet)
    critic_val1 = Critic_val() # imnet)
    log_alpha = tf.Variable(0.0)
    alpha = tf.math.exp(log_alpha)
    target_entropy = 0.4#-np.prod(action_dim)
    optimizer_alp = optimizers.Adam(learning_rate = 1e-6)
    lr_schedual = optimizers.schedules.ExponentialDecay(2.5e-4,10000,0.98,staircase=True)
    optimizer_cri = mixed_precision.LossScaleOptimizer(optimizers.Adam(learning_rate = lr_schedual,epsilon=1e-5))
    optimizer_ac = mixed_precision.LossScaleOptimizer(optimizers.Adam(learning_rate = lr_schedual,epsilon=1e-5))

    trainNet(True,render)
    env.close()

