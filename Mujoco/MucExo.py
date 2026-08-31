import gymnasium as gym
import tensorflow as tf
from keras import losses, layers
from  keras import optimizers
import random
import numpy as np
import os 
from MuagentImEp import  Actor_val,Critic_val
import datetime
from keras import mixed_precision
# import tensorflow_probability.python.distributions as tfd
# import envpool
from Memo import Memory, kl_normal, action_pro, VectorEnvNormObs
import copy as cp
from collections import deque
import tensorflow_probability.python.distributions as tfp
os.environ["KERAS_BACKEND"] = "tensorflow"
from numba import njit

'Base: Entropy -  pi * V * pi/beta  // V -> reward ,no clip + kl (instant)'
'deque for off-memory'

'my algo. 没有向 Value Function 传递 action'

os.environ['CUDA_VISIBLE_DEVICES']='0'
ENV_NAME = 'Humanoid-v5'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
GAMMA = 0.990 # 未来奖励的衰减
Replay_N = 4 # 观测存储器D的容量
BATCH = 256 # 训练batch大小
Run_Step = 256
Train_Env_num = 2
Test_Env_num = 4
Train_step = 2 # 8:4
OBSERVE = Train_Env_num * Replay_N * Run_Step -1  #* Run_Step // 256 训练前观察积累的轮数
beta = 5
klra = 0.8
EPS = 2480 * 2 * 5 // Train_Env_num 
Rec_Num = 16 * 4  // Train_Env_num
Rec_step = Train_Env_num * Run_Step * Rec_Num
All_step = Train_Env_num * Run_Step * EPS

@njit
def sample_inde():
    indices = np.random.permutation(Run_Step * Train_Env_num * Replay_N)
    for idx in range(0, indices.size, BATCH):
        yield indices[idx:idx + BATCH]

def evaluate():
    s = eva_env.reset()[0].astype(dtype=np.float32)
    rew = np.zeros(Test_Env_num)
    rew_n = np.zeros(Test_Env_num)
    count = 0
    done_envs = []
    while count < Test_Env_num:
        xt, logsigma = actor_val(s)
        # dist = tfd.Normal(xt,0.5+np.zeros_like(xt))
        action, _ = action_pro(xt,logsigma) # tf.squeeze(dist.sample([1]),0)
        action_ = action_bound * tf.tanh(action).numpy() # tf.clip_by_value(action,-1,1).numpy() 
        # print(action)
        s,r,d,ter,_= eva_env.step(action_)
        # r = np.where(ter==True, r-10., r)
        d = np.logical_or(d,ter)
        rew += r 
        if np.any(d) :
            env_end = np.where(d)[0]
            # s[env_end] = eva_env.reset(env_end)[0].astype(dtype=np.float32)
            for i in env_end:
                if i in done_envs:
                    continue
                done_envs.append(i)
                rew_n[count] = rew[i]
                count += 1
                if count == Test_Env_num:
                    break
            rew[env_end]=0
        s = s.astype(dtype=np.float32)
    print(logsigma.numpy().mean())
    return rew_n.mean(), rew_n.std()

def trainNet(istrain):
    # 创建网络
    # tf.random.set_seed(42)

    # 将每一轮的观测存在D中，之后训练从D中随机抽取batch个数据训练，以打破时间连续导致的相关性，保证神经网络训练所需的随机性。
    mem = [Memory(state_dim, action_dim, Run_Step) for _ in range(Train_Env_num)]  # Memory
    D_mem:deque[Memory] = deque(maxlen = Replay_N * Train_Env_num)

    t=0
    # temp_t=0
    s=env.reset()[0].astype(dtype=np.float32)
    ep_reward = 0
    ep_best=0

    # tensorboard
    train_log_dir='logs/'+ENV_NAME+'/'+datetime.datetime.now().strftime("%m%d-%H%M%S")+'beta:'+str(beta)+str(klra)+'re:'+str(Replay_N)+'entro'
    train_sum_writer = tf.summary.create_file_writer(train_log_dir)

    eps=0 #
    while eps < EPS:
        for _ in range(Run_Step):
            xt, logSigma = actor_val(s)
            action, log_probs = action_pro(xt, logSigma)
            action_ = action_bound * tf.tanh(action).numpy() #tf.clip_by_value(action,-1,1).numpy()
            s_t,r,d,ter,_= env.step(action_)
            # np.where(ter==True,r-10.,r)
            d = np.logical_or(d,ter)
            # if np.any(d) :
            #     env_end = np.where(d)[0]
            #     s_t[env_end] = env.reset(env_end)[0].astype(dtype=np.float32)
            for i in range(Train_Env_num):
                mem[i].add(s[i],np.stack((xt[i],logSigma[i])),r[i],int(d[i]),log_probs[i],action[i])
            s = s_t.astype(dtype=np.float32)
        t += Train_Env_num * Run_Step
        for i in range(Train_Env_num):
            mem[i].add(s[i], None, None, None, None, None)
            mem[i].compute_gae(critic_val)
        D_mem.extend(cp.deepcopy(mem))
        
#============================ 训练网络 ===========================================
        # 观测一定轮数后开始训练
        if t > OBSERVE and istrain : # 
            # 随机抽取minibatch个数据训练
            eps+=1
            obses = np.concatenate([i.obses[:-1] for i in D_mem],0)
            discounted_rew_sum = np.concatenate([i.discounted_rew_sum for i in D_mem],0)
            policy = np.concatenate([i.policy for i in D_mem],0)
            gae = np.concatenate([i.gae for i in D_mem],0)
            actions = np.concatenate([i.actions for i in D_mem],0)
            means = np.concatenate([i.means for i in D_mem],0)
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
                    # b_r = discounted_rew_sum[minibatch]
                    # b_r = (b_r - b_r.mean())/(b_r.std()+1e-8)
                    # b_r = tf.convert_to_tensor(b_r,dtype=tf.float32)
                    b_mean = tf.convert_to_tensor(means[minibatch],dtype=tf.float32)
                    b_prob = tf.convert_to_tensor(policy[minibatch],dtype=tf.float32) # probability
                    b_gae = tf.convert_to_tensor(gae[minibatch],dtype=tf.float32)
                    # b_gae = gae[minibatch]
                    # b_gae = (b_gae - b_gae.mean())/(b_gae.std()+1e-8)
                    # b_gae = tf.convert_to_tensor(b_gae,dtype=tf.float32)
                    loss1 = train_critic(b_r,b_s)
                    ac_loss = train_actor(b_s,b_mean,b_prob,b_gae,b_a)
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
                # print(b_gae.numpy().mean())
                print("ep=",eps,"loss1 = %f" % loss1,"ac-loss = %f" % ac_loss,"score=",ep_reward,'+',ep_std)

                    # tensorboard
                        
                with train_sum_writer.as_default():
                    tf.summary.scalar('rate',ac_loss,step = t)
                    tf.summary.scalar('std',ep_std,step = t)
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
    with tf.GradientTape() as tape:
        dq1 = tf.squeeze(critic_val(b_s))
        loss = losses.huber(y,dq1) 
        # loss = optimizer_cri.get_scaled_loss(loss)
        # print("loss1 = %f " % loss1)
    gradients = tape.gradient(loss, critic_val.trainable_variables)
    # gradients = optimizer_cri.get_unscaled_gradients(gradients)
    optimizer_cri.apply(gradients, critic_val.trainable_variables)
    return loss

@tf.function
def train_actor(b_s,b_means,b_prob,b_gae,b_a):
    batchdist = tfp.Independent(tfp.Normal(b_means[:,0], b_means[:,1]),reinterpreted_batch_ndims=1)
    with tf.GradientTape() as tape: #在这个空间里面计算梯度
        xt, Sigma = actor_val(b_s)
        # Sigma = tf.math.exp(logSigma)
        dist = tfp.Independent(tfp.Normal(xt, Sigma),reinterpreted_batch_ndims=1)
        log_probs = dist.log_prob(b_a)
        # log_probs = tf.reduce_sum(tf.math.log(tf.math.exp(-0.5*tf.square(b_a-xt)/tf.math.square(Sigma))/(tf.math.sqrt(2*np.pi)*Sigma)),-1)
        dq1 = tf.math.exp(log_probs - b_prob) # rate = pi/beta
        # Entr = alpha*log_probs # Entropy 
        dq0 = 1
        dq2 = tf.where(dq1 >  0.2 + dq0,  0.2 + dq0 + 1/beta - tf.exp(beta * (0.2+dq0-dq1))/beta, dq1) # 需要先算大的(-dq1)的部分, 避免exp数值太大
        dq2 = tf.where(dq2 < -0.2 + dq0, -0.2 + dq0 - 1/beta + tf.exp(beta * (dq2-(dq0-0.2)))/beta, dq2)
        Q = - tf.minimum(tf.multiply(dq1, b_gae),tf.multiply(dq2,b_gae)) # dq0+  # pi * (Entropy -Q) * rate -0.01*Entr
        # kl1 = kl_normal(xt,b_means,Sigma)
        kl1 = klra * tfp.kl_divergence(batchdist, dist) # 0.01 *+ 0.01 * dist.entropy()
        ac_loss = tf.reduce_mean(Q + kl1) 
    gradients = tape.gradient(ac_loss, actor_val.trainable_variables)
    # gradients = optimizer_ac.get_unscaled_gradients(gradients)
    optimizer_ac.apply(gradients, actor_val.trainable_variables)
    dq1 = tf.reduce_mean(dq1)#tf.math.abs(tf.math.log(dq1+1e-6)))
    return ac_loss


# @tf.function
# def train_alpha(b_s):
#     a_temp = actor_val(b_s)
#     log_pis = tf.math.log(a_temp + 1e-8)
#     with tf.GradientTape() as tape:
#         loss = -tf.reduce_mean(log_alpha*(log_pis+target_entropy))
#     gradients = tape.gradient(loss, [log_alpha])
#     optimizer_alp.apply_gradients(zip(gradients, [log_alpha]))
#     return loss

if __name__ == "__main__":
    seed = 0
    # env =  envpool.make_gymnasium(ENV_NAME,num_envs=Train_Env_num,seed=seed)# gym.make(ENV_NAME)
    # eva_env = envpool.make_gymnasium(ENV_NAME,num_envs=Test_Env_num,seed=seed)
    env = VectorEnvNormObs(gym.make_vec(ENV_NAME,num_envs=Train_Env_num))
    eva_env = VectorEnvNormObs(gym.make_vec(ENV_NAME,num_envs=Test_Env_num),update_obs_rms=False)
    eva_env.set_obs_rms(env.get_obs_rms())
    policy = mixed_precision.Policy('float32')
    mixed_precision.set_global_policy(policy)
    state_dim = env.observation_space.shape[1]
    action_dim = env.action_space.shape[1]
    action_bound = env.action_space.high[0]
    # action_dim=4
    # imnet = ImagePro()
    actor_val = Actor_val(action_dim) #, imnet)
    critic_val = Critic_val() # imnet)

    lr_schedual = optimizers.schedules.ExponentialDecay(2.5e-4,5000,0.98,staircase=True)
    optimizer_ac = optimizers.Adam(learning_rate = lr_schedual,epsilon=1e-8) # mixed_precision.LossScaleOptimizer()
    lr_schedual = optimizers.schedules.ExponentialDecay(2e-4,5000,0.98,staircase=True)
    optimizer_cri = optimizers.Adam(learning_rate = lr_schedual,epsilon=1e-8) # mixed_precision.LossScaleOptimizer()

    trainNet(True)
    env.close()
