from MuagentImEp import  Critic_val,Actor_val
import tensorflow as tf
import numpy as np
from typing import Union, Optional

 # 未来奖励的衰减

class Memory:
    def __init__(self, obs_shape, action_dim, Run_Step = 256, Gamma = 0.990):
        self.rms = RunningMeanStd(epsilon=1e-8)
        self.size = Run_Step
        self.obses   = np.zeros((self.size+1, obs_shape),dtype=np.float32)
        self.actions = np.zeros((self.size+1, action_dim),dtype=np.float32)
        self.means   = np.zeros((self.size,action_dim),dtype=np.float32) # 存储神经网络的输出,用于计算kl散度
        self.rewards = np.zeros((self.size),dtype=np.float32)
        self.dones   = np.zeros((self.size),dtype=np.int32)
        # self.values  = np.zeros((self.size),dtype=np.float32)
        self.policy  = np.zeros((self.size,action_dim),dtype=np.float32)
        self.deltas  = np.zeros((self.size),dtype=np.float64)
        self.discounted_rew_sum = np.zeros((self.size),dtype=np.float32)
        self.gae = np.zeros((self.size),dtype=np.float32)
        self.sample_i = 0  
        self.Gamma = Gamma

    def __len__(self):
        return self.size

    def add(self, obs, xt, reward, done, policy, action):
        """
        Adds a new data sample to the replay buffer.

        Parameters:
            obs (object): The observation of the environment.
            action (object): The action taken in the environment.
            reward (float): The reward received from the environment.
            done (bool): Whether the episode is done or not.
            policy (float): The probability of the action taken.

        Returns:
            None
        """
        if self.sample_i < self.size:
            self.rewards[self.sample_i] = reward
            self.dones[self.sample_i] = done
            self.policy[self.sample_i] = policy
            self.means[self.sample_i] = xt
        self.actions[self.sample_i] = action
        self.obses[self.sample_i] = obs
        self.sample_i += 1

    def compute_gae(self, net_cri:Critic_val):
        values = net_cri(tf.convert_to_tensor(self.obses,tf.float32),tf.convert_to_tensor(self.actions,tf.float32))
        # values = tf.gather_nd(values,self.actions,batch_dims=1)
        values = tf.squeeze(values).numpy()
        values_t = np.roll(values,-1)[:self.size]
        values = values[:self.size]
        values   = values   * np.sqrt(self.rms.var + self.rms.eps)
        values_t = values_t * np.sqrt(self.rms.var + self.rms.eps)
        self.deltas = self.rewards + self.Gamma * values_t * (1-self.dones) - values
        self.gae[-1] = self.deltas[-1]
        for t in reversed(range(self.size-1)):
            self.gae[t] = self.deltas[t] + (1 - self.dones[t]) * (self.Gamma * 0.95) * self.gae[t + 1]
        discounted_rew_sum = self.gae + values
        self.discounted_rew_sum = discounted_rew_sum / np.sqrt(self.rms.var + self.rms.eps)
        self.rms.update(discounted_rew_sum)
        # self.discounted_rew_sum = ( - np.mean(self.discounted_rew_sum)) / (np.std(self.discounted_rew_sum) + 1e-8) 
        return

    def reset(self):
        self.sample_i = 0
        self.obses = np.zeros_like(self.obses)
        self.rewards = np.zeros_like(self.rewards)
        # self.values = np.zeros_like(self.values)
        self.policy = np.zeros_like(self.policy)
        self.deltas = np.zeros_like(self.deltas)
        self.discounted_rew_sum = np.zeros_like(self.discounted_rew_sum)
        self.gae = np.zeros_like(self.gae)
        self.actions = np.zeros_like(self.actions)
        self.means = np.zeros_like(self.means)

class RunningMeanStd(object):
    """Calculates the running mean and std of a data stream.

    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    :param mean: the initial mean estimation for data array. Default to 0.
    :param std: the initial standard error estimation for data array. Default to 1.
    :param float clip_max: the maximum absolute value for data array. Default to
        10.0.
    :param float epsilon: To avoid division by zero.
    """

    def __init__(
        self,
        mean: Union[float, np.ndarray] = 0.0,
        std: Union[float, np.ndarray] = 1.0,
        clip_max: Optional[float] = 10.0,
        epsilon: float = np.finfo(np.float32).eps.item(),
    ) -> None:
        self.mean, self.var = mean, std
        self.clip_max = clip_max
        self.count = 0
        self.eps = epsilon

    def norm(self, data_array: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        data_array = (data_array - self.mean) / np.sqrt(self.var + self.eps)
        if self.clip_max:
            data_array = np.clip(data_array, -self.clip_max, self.clip_max)
        return data_array

    def update(self, data_array: np.ndarray) -> None:
        """Add a batch of item into RMS with the same shape, modify mean/var/count."""
        batch_mean, batch_var = np.mean(data_array, axis=0), np.var(data_array, axis=0)
        batch_count = len(data_array)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean, self.var = new_mean, new_var
        self.count = total_count

def kl_normal(u1, u2, sig1, sig2):
    """
    Calculates the Kullback-Leibler divergence between two normal distributions.

    Args:
        u1 (tf.Tensor): The mean of the first distribution.
        u2 (tf.Tensor): The mean of the second distribution.
        sig1 (tf.Tensor): The standard deviation of the first distribution.
        sig2 (tf.Tensor): The standard deviation of the second distribution.

    Returns:
        tf.Tensor: The Kullback-Leibler divergence between the two distributions.
    """
    num = tf.math.square(u1-u2) + tf.math.square(sig1)-tf.math.square(sig2)
    den = 2 * tf.math.square(sig2) + 1e-8
    y = num / den + tf.math.log(sig2) - tf.math.log(sig1)
    return tf.reduce_sum(y,-1)

def action_pro(mean:tf.Tensor,Sigma):
    action = mean + tf.random.normal(shape=mean.shape, dtype=tf.float32) * Sigma
    log_prob = tf.math.log(tf.math.exp(-0.5*tf.square(action - mean)/tf.math.square(Sigma))/(tf.math.sqrt(2*np.pi)*Sigma))
    return action,log_prob
