import gymnasium as gym
import tensorflow as tf
from keras import losses
import keras.optimizers as optimizers
import random
import numpy as np
from collections import deque
import os 
from AtariagentIm import  Actor_val,Critic_val,ImagePro
import datetime
# import platform
# from ReplayBuffer import ReplayBuffer
import keras.mixed_precision  as mixed_precision

' Base: Entropy -  pi * V * clip(pi/beta)  // V -> reward'

os.environ['CUDA_VISIBLE_DEVICES']='0'
ENV_NAME = 'ALE/Breakout-v5'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20000)])

GAMMA = 0.990 # 未来奖励的衰减
 # 训练前观察积累的轮数
REPLAY_MEMORY = 10000 # 观测存储器D的容量
BATCH = 128 # 训练batch大小
OBSERVE = BATCH + 1000
TAU=0.995
Run_Step = 4096

class Memory:
    def __init__(self, obs_shape):
        self.size = Run_Step
        self.obses   = np.zeros((self.size+1, 210,160,3),dtype=np.float32)
        self.actions = np.zeros((self.size),dtype=np.int32)
        self.rewards = np.zeros((self.size),dtype=np.float32)
        self.dones   = np.zeros((self.size),dtype=np.int32)
        # self.values  = np.zeros((self.size),dtype=np.float32)
        self.policy  = np.zeros((self.size),dtype=np.float32)
        self.deltas  = np.zeros((self.size),dtype=np.float32)
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
        self.gae = (self.gae - np.mean(self.gae)) / (np.std(self.gae) + 1e-8) 
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
    rew = 0.
    while True:
        s = tf.expand_dims(tf.convert_to_tensor(s, dtype=tf.float32),0)
        action = actor_val(s)[0]
        action = int(np.random.choice(action_dim,1,p=action.numpy()))
        s,r,d,ter,_= eva_env.step(action)
        # if ter:
            # r = -10.
        rew += r
        if d or ter :
            break
    return rew

def trainNet(istrain, isrender):
    # 创建网络
    # tf.random.set_seed(42)

    # 将每一轮的观测存在D中，之后训练从D中随机抽取batch个数据训练，以打破时间连续导致的相关性，保证神经网络训练所需的随机性。
    mem = Memory(state_dim)  # Memory
    # D_mem = deque(maxlen=REPLAY_MEMORY)

    t = 0
    s = env.reset()[0].astype(dtype=np.float32)
    ep_reward = 0
    ep_best=0

    # tensorboard
    # train_log_dir='logs/Qtest-KL-1121/'+datetime.datetime.now().strftime("%m%d-%H-%M")+'Qmax-mean'
    # train_sum_writer = tf.summary.create_file_writer(train_log_dir)

    eps=0
    while eps < 150000:
        for _ in range(Run_Step):
            t += 1
            if isrender:
                env.render() # 显示
        
            stemp = tf.expand_dims(s,0)
            xt = actor_val(stemp)[0]
            action = int(np.random.choice(action_dim,1,p=xt.numpy()))
            s_t,r,d,ter,_= env.step(action)
            if ter:
                d = True
            mem.add(s,action,r,int(d),xt[action])
            s = s_t.astype(dtype=np.float32)
            if d :
                s = env.reset()[0].astype(dtype=np.float32) # 完成后reset
                ep_reward=0
        mem.add(s, None, None, None, None)
        mem.compute_gae(critic_val1)
        # for i in range(Run_Step):
        #     D_mem.append((mem.obses[i],[mem.actions[i]],mem.discounted_rew_sum[i],mem.dones[i],mem.policy[i],mem.gae[i]))

#============================ 训练网络 ===========================================
        # 观测一定轮数后开始训练
        if istrain :
            inde = list(range(Run_Step))
            random.shuffle(inde)
            # 随机抽取minibatch个数据训练
            eps+=1
            for i in range(64):
                minibatch = random.sample(inde, BATCH)
                # print("==================start train====================t=",t)

                # minibatch = mem.sample_minibatch(BATCH)
                # 获得batch中的每一个变量
                b_s = tf.convert_to_tensor(mem.obses[minibatch])
                b_a = tf.expand_dims(tf.convert_to_tensor(mem.actions[minibatch]),-1)
                b_r = tf.convert_to_tensor(mem.discounted_rew_sum[minibatch],dtype=tf.float32)
                # b_s_ = tf.convert_to_tensor([d[3] for d in minibatch])
                # b_done = tf.convert_to_tensor([d[3] for d in minibatch],dtype=tf.float32)
                b_ra = tf.convert_to_tensor(mem.policy[minibatch],dtype=tf.float32)
                b_gae = tf.convert_to_tensor(mem.gae[minibatch],dtype=tf.float32)
                loss1 = train_critic1(b_r,b_s)
                # 更新actor
                ac_loss = train_actor(b_s,b_a,b_ra,b_gae)
                # train_kl(b_s)
                train_alpha(b_s)
                alpha = tf.math.exp(log_alpha)
                # soft: ExponentialMovingAverage更新参数方法
                # actor_tar.set_weights(actor_val.get_weights())
                    
                # if i == temp_t-1:
            ep_reward = evaluate()
            print("ep=",eps,"loss1 = %f" % loss1,"ac-loss = %f" % ac_loss,"score=",ep_reward)

                # tensorboard
                    
            # with train_sum_writer.as_default():
            # #     tf.summary.scalar('cri-loss1',loss1,step=eps)
            # #     tf.summary.scalar('cri-loss2',loss2,step=eps)
            #     tf.summary.scalar('ac-loss',ac_loss,step=eps)
            #     tf.summary.scalar('score',ep_reward,step=eps)            
            
            # t=0
            # 每1000轮保存一次网络参数
            if  ep_best < ep_reward or ep_reward > 200:
                ep_best = ep_reward
                print("=================model save====================")

        mem.reset()



@tf.function
def train_critic1(y,b_s):
    # 训练Critic
    # acts_dim = tf.expand_dims(tf.argmax(b_a,1),1)
    # b_a = actor_val(b_s)+tf.clip_by_value(tf.random.normal([BATCH,action_dim],0,0.1,tf.float32), -0.5,0.5)
    with tf.GradientTape() as tape:
        dq1 = tf.squeeze(critic_val1(b_s))
        loss1 = losses.MAE(y,dq1) 
        loss = optimizer_ac1.get_scaled_loss(loss1)
        # print("loss1 = %f " % loss1)
    gradients = tape.gradient(loss, critic_val1.trainable_variables)
    gradients = optimizer_ac1.get_unscaled_gradients(gradients)
    optimizer_ac1.apply_gradients(zip(gradients, critic_val1.trainable_variables))
    return loss1

@tf.function
def train_actor(b_s,b_a,b_ra,b_gae):
    with tf.GradientTape() as tape: #在这个空间里面计算梯
        a_temp = actor_val(b_s)
        log_pis = tf.math.log(a_temp + 1e-8)
        # dq1 = tf.math.minimum(critic_val1(b_s),critic_val2(b_s))
        # dq1 = tf.gather_nd(dq1,b_a,batch_dims=1)
        dq2 = tf.divide(tf.gather_nd(a_temp,b_a,batch_dims=1),b_ra + 1e-8) # rate = pi/beta
        clip_adv = tf.where(b_gae > 0, 1.2*b_gae, 0.8*b_gae)
        # Q = tf.reduce_sum(tf.multiply(a_temp,alpha*log_pis),-1) - tf.multiply(dq2,dq1)
        Q = tf.reduce_sum(tf.multiply(a_temp,alpha*log_pis),-1) # Entropy 
        dq2 = tf.minimum(tf.multiply(dq2, b_gae),clip_adv) # clip
        Q = 0.01*Q-dq2  # pi * (Entropy -Q) * rate
        ac_loss = tf.reduce_mean(Q)
        ac_loss1 = optimizer_ac3.get_scaled_loss(ac_loss)
        # print("ac-loss = %f" % ac_loss)
    gradients = tape.gradient(ac_loss1, actor_val.trainable_variables)
    gradients = optimizer_ac3.get_unscaled_gradients(gradients)
    optimizer_ac3.apply_gradients(zip(gradients, actor_val.trainable_variables))
    return ac_loss

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
    env = gym.make(ENV_NAME)
    eva_env = gym.make(ENV_NAME)
    policy = mixed_precision.Policy('float32')
    mixed_precision.set_global_policy(policy)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    # action_bound = 
    # action_dim=4
    imnet = ImagePro()
    actor_val = Actor_val(action_dim, imnet)
    critic_val1 = Critic_val(imnet)
    log_alpha = tf.Variable(0.0)
    alpha = tf.math.exp(log_alpha)
    target_entropy = 0.4#-np.prod(action_dim)
    optimizer_alp = optimizers.Adam(learning_rate = 2e-5)
    optimizer_ac1 = mixed_precision.LossScaleOptimizer(optimizers.Adam(learning_rate = 1e-3))
    optimizer_ac3 = mixed_precision.LossScaleOptimizer(optimizers.Adam(learning_rate = 1e-4))
    optimizer_ac4 = mixed_precision.LossScaleOptimizer(optimizers.Adam(learning_rate = 5e-4))

    trainNet(True,render)
    env.close()

