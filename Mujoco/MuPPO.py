import gymnasium as gym
import tensorflow as tf
from keras import losses
from keras import optimizers
import random
import numpy as np
import os 
from MuagentImEp import  Actor_val,Critic_val
import datetime
from keras import mixed_precision
import tensorflow_probability.python.distributions as tfd
# import envpool
from Memo import Memory, action_pro, VectorEnvNormObs
'EnvPool'

'value clip, adv clip, adv norm, PPO'

os.environ['CUDA_VISIBLE_DEVICES']='1'
ENV_NAME = 'Humanoid-v5'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])

GAMMA = 0.990 # 未来奖励的衰减
# REPLAY_MEMORY = 8192 # 观测存储器D的容量
BATCH = 256 # 训练batch大小
Run_Step = 256
Train_Env_num = 8
Test_Env_num = 4
Train_step = 4 #* Train_Env_num
# Sigma = 0.6
EPS = 2480 * 2 // Train_Env_num 
Rec_Num = 8 # * 8 // Train_Env_num
Rec_step = Train_Env_num * Run_Step * Rec_Num
All_step = Train_Env_num * Run_Step * EPS

def sample_inde():
    indices = np.random.permutation(Run_Step * Train_Env_num)
    for idx in range(0, indices.size, BATCH):
        yield indices[idx:idx + BATCH]

def evaluate():
    s = eva_env.reset()[0].astype(dtype=np.float32)
    rew = np.zeros(Test_Env_num)
    rew_n = np.zeros(Test_Env_num)
    count = 0
    done_envs = []
    # sigma = 0.01
    while count < Test_Env_num:
        xt,sigma = actor_val(s)
        # dist = tfd.Normal(xt,0.5+np.zeros_like(xt))
        action,_ = action_pro(xt,sigma) # tf.squeeze(dist.sample([1]),0)tf.clip_by_value(action,-1,1).numpy()
        action = action_bound * tf.tanh(action).numpy() # tf.clip_by_value(action,-1,1).numpy()
        # print(action)
        s,r,d,ter,_= eva_env.step(action)
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
    # print(xt.numpy().mean())
    print(sigma.numpy().mean())
    return rew_n.mean(), rew_n.std()


def trainNet(istrain):
    # 将每一轮的观测存在D中，之后训练从D中随机抽取batch个数据训练，以打破时间连续导致的相关性，保证神经网络训练所需的随机性。
    mem = [Memory(state_dim, action_dim, Run_Step) for _ in range(Train_Env_num)]  # Memory

    t = 0
    s = env.reset()[0].astype(dtype=np.float32)
    ep_reward = 0
    ep_best=0
    # tensorboard
    train_log_dir='logs/'+ENV_NAME+'/'+datetime.datetime.now().strftime("%m%d-%H%M%S")+'PPO'
    train_sum_writer = tf.summary.create_file_writer(train_log_dir)

    eps=0
    while eps < EPS:
        # if eps < 2500:
        #     Sigma = 0.6 - eps*2e-4
        # else:
        #     Sigma = 0.1
        for _ in range(Run_Step):
            xt, logSigma = actor_val(s)
            action, log_probs = action_pro(xt,logSigma)
            # log_probs = tf.reduce_sum(log_probs,-1)# - tf.reduce_sum(tf.math.log(1 - tf.math.square(action) + 1e-6),-1)
            action_ = action_bound * tf.tanh(action).numpy() #tf.clip_by_value(action,-1,1).numpy()
            s_t,r,d,ter,_= env.step(action_)
            # r = np.where(ter==True, r-10., r)
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
        # for i in range(Run_Step):
        #     D_mem.append((mem.obses[i],[mem.actions[i]],mem.discounted_rew_sum[i],mem.dones[i],mem.policy[i],mem.gae[i]))

#============================ 训练网络 ===========================================
        # 观测一定轮数后开始训练
        if istrain :
            # 随机抽取minibatch个数据训练
            eps+=1
            obses = np.concatenate([i.obses[:-1] for i in mem],0)
            discounted_rew_sum = np.concatenate([i.discounted_rew_sum for i in mem],0)
            policy = np.concatenate([i.policy for i in mem],0)
            gae = np.concatenate([i.gae for i in mem],0)
            actions = np.concatenate([i.actions for i in mem],0) 
            for i in range(Train_step):
                # j = i % Train_Env_num
                for minibatch in sample_inde():

                # 获得batch中的每一个变量
                    b_s = tf.convert_to_tensor(obses[minibatch])
                    b_a = tf.convert_to_tensor(actions[minibatch])
                    b_r = tf.convert_to_tensor(discounted_rew_sum[minibatch],dtype=tf.float32)
                    # b_r = discounted_rew_sum[minibatch]
                    # b_r = (b_r - b_r.mean())/(b_r.std()+1e-8)
                    # b_r = tf.convert_to_tensor(b_r,dtype=tf.float32)
                    b_prob = tf.convert_to_tensor(policy[minibatch],dtype=tf.float32) # probability
                    b_gae = tf.convert_to_tensor(gae[minibatch],dtype=tf.float32)
                    # b_gae = gae[minibatch]
                    # b_gae = (b_gae - b_gae.mean())/(b_gae.std()+1e-8)
                    # b_gae = tf.convert_to_tensor(b_gae,dtype=tf.float32)
                    loss1 = train_critic(b_r,b_s)
                    # 更新actor
                    ac_loss = train_actor(b_s,b_prob,b_gae,b_a)

            if eps % Rec_Num == 1:  # 一个 eps = 1024 step
                ep_reward, ep_std = evaluate()
                print("ep=",eps,"loss1 = %f" % loss1,"ac-loss = %f" % ac_loss,"score=",ep_reward,'+',ep_std)
                # tensorboard  ,"lr=",optimizer_ac.learning_rate
                with train_sum_writer.as_default():
                    tf.summary.scalar('rate',ac_loss,step = t)
                    tf.summary.scalar('std',ep_std,step = t)
                    tf.summary.scalar('score',ep_reward,step = t)
            
            # t=0
            # 每1000轮保存一次网络参数
                if  ep_best < ep_reward:
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
        loss = losses.huber(y,dq1) 
        # loss = optimizer_cri.get_scaled_loss(loss1)
        # print("loss1 = %f " % loss1)
    gradients = tape.gradient(loss, critic_val.trainable_variables)
    # gradients = optimizer_cri.get_unscaled_gradients(gradients)
    optimizer_cri.apply_gradients(zip(gradients, critic_val.trainable_variables))
    return loss

# @tf.function
def train_actor(b_s,b_prob,b_gae,b_a):
    with tf.GradientTape() as tape: #在这个空间里面计算梯
        xt, Sigma = actor_val(b_s)
        # Sigma = tf.math.exp(logSigma)
        dist = tfd.Independent(tfd.Normal(xt, Sigma),reinterpreted_batch_ndims=1)
        log_probs = dist.log_prob(b_a)
        # log_probs = tf.reduce_sum(tf.math.log(tf.math.exp(-0.5*tf.square(b_a-xt)/tf.math.square(Sigma))/(tf.math.sqrt(2*np.pi)*Sigma)),-1)
        dq1 = tf.math.exp(log_probs - b_prob) # rate = pi/beta
        # Entr = tf.reduce_mean(tf.math.log(logSigma),-1) # Entropy   0.5 + 
        # clip_adv = tf.where(b_gae > 0, 1.2 *b_gae, 0.8*b_gae)
        # dq2 = tf.minimum(tf.multiply(dq1, b_gae),clip_adv)
        cliped_dq = tf.clip_by_value(dq1, 0.8, 1.2)
        dq2 = tf.minimum(tf.multiply(dq1, b_gae),tf.multiply(cliped_dq, b_gae))
        Q = - dq2 # 0.1*Entr   kl_normal(a_temp[0],b_para[:,:,0],a_temp[1],b_para[:,:,1]) # pi * (Entropy -Q) * rate -0.01*Entr
        ac_loss = tf.reduce_mean(Q)
        # ac_loss1 = optimizer_ac.get_scaled_loss(ac_loss)
        # print("ac-loss = %f" % ac_loss)
    gradients = tape.gradient(ac_loss, actor_val.trainable_variables)
    # gradients = optimizer_ac.get_unscaled_gradients(gradients)
    optimizer_ac.apply(gradients, actor_val.trainable_variables)
    dq1 = tf.reduce_mean(dq1) # tf.math.abs(tf.math.log(dq1+1e-6)))
    return ac_loss


if __name__ == "__main__":
    seed = 0
    # env =  envpool.make_gymnasium(ENV_NAME,num_envs=Train_Env_num,seed=seed)# gym.make(ENV_NAME)
    # eva_env = envpool.make_gymnasium(ENV_NAME,num_envs=Test_Env_num,seed=seed)
    env = VectorEnvNormObs(gym.make_vec(ENV_NAME,num_envs=Train_Env_num)) # gym.make(), render_mode="rgb_array""MountainCarContinuous-v0", goal_velocity=0.1
    eva_env = VectorEnvNormObs(gym.make_vec(ENV_NAME,num_envs=Test_Env_num),update_obs_rms=False) # "MountainCarContinuous-v0", goal_velocity=0.1
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


    lr_schedual = optimizers.schedules.ExponentialDecay(2.5e-4,5000,0.99,staircase=True)
    optimizer_ac = mixed_precision.LossScaleOptimizer(optimizers.Adam(learning_rate = lr_schedual,epsilon=1e-8)) #
    lr_schedual = optimizers.schedules.ExponentialDecay(2e-4,5000,0.99,staircase=True)
    optimizer_cri = mixed_precision.LossScaleOptimizer(optimizers.Adam(learning_rate = lr_schedual,epsilon=1e-8))

    trainNet(True)
    env.close()

