import gymnasium as gym
import numpy as np
import tensorflow as tf
from keras import layers,optimizers

import tensorflow_probability.python.distributions as tfp
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])

# 超参数设置
env_name = 'Ant-v5'
gamma = 0.99
gae_lambda = 0.95
clip_ratio = 0.2
epochs = 10
batch_size = 64
policy_lr = 3e-4
value_lr = 1e-3
train_policy_iters = 80
train_value_iters = 80
max_steps = 2048
total_episodes = 1000

# 创建环境
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

class PolicyNetwork(tf.keras.Model):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.dense1 = layers.Dense(64, activation='tanh')
        self.dense2 = layers.Dense(64, activation='tanh')
        self.mean = layers.Dense(action_dim)
        self.log_std = tf.Variable(tf.zeros(action_dim), trainable=True)

    def call(self, states):
        x = self.dense1(states)
        x = self.dense2(x)
        return self.mean(x)

    def get_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        mean = self.call(state)
        std = tf.exp(self.log_std)
        dist = tfp.Normal(mean, std)
        action = dist.sample()
        log_prob = tf.reduce_sum(dist.log_prob(action), axis=-1)
        return action.numpy()[0], log_prob.numpy()[0]

class ValueNetwork(tf.keras.Model):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.dense1 = layers.Dense(64, activation='tanh')
        self.dense2 = layers.Dense(64, activation='tanh')
        self.value = layers.Dense(1)

    def call(self, states):
        x = self.dense1(states)
        x = self.dense2(x)
        return self.value(x)

class PPOAgent:
    def __init__(self):
        self.policy_net = PolicyNetwork()
        self.value_net = ValueNetwork()
        self.policy_optimizer = optimizers.Adam(learning_rate=policy_lr)
        self.value_optimizer = optimizers.Adam(learning_rate=value_lr)

    def collect_experience(self, env: gym.Env, max_steps):
        states, actions, log_probs, rewards, dones = [], [], [], [], []
        state = env.reset()[0]
        for _ in range(max_steps):
            action, log_prob = self.policy_net.get_action(state)
            next_state, reward, done, ter, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done or ter)
            
            state = next_state
            if done or ter:
                state = env.reset()[0]
                
        return (
            np.array(states),
            np.array(actions),
            np.array(log_probs),
            np.array(rewards),
            np.array(dones)
        )

    def compute_advantages(self, rewards, dones, states):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        values = self.value_net(states).numpy().flatten()
        next_values = np.append(values[1:], 0)
        
        deltas = rewards + gamma * next_values * (1 - dones) - values
        advantages = np.zeros_like(rewards)
        advantage = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                advantage = 0
            advantage = deltas[t] + gamma * gae_lambda * (1 - dones[t]) * advantage
            advantages[t] = advantage
        returns = advantages + values
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return advantages, returns

    def train_policy(self, states, actions, log_probs_old, advantages):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        log_probs_old = tf.convert_to_tensor(log_probs_old, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)

        for _ in range(train_policy_iters):
            indices = tf.range(states.shape[0])
            indices = tf.random.shuffle(indices)
            for start in range(0, states.shape[0], batch_size):
                end = start + batch_size
                idx = indices[start:end]
                batch_states = tf.gather(states, idx)
                batch_actions = tf.gather(actions, idx)
                batch_log_probs_old = tf.gather(log_probs_old, idx)
                batch_advantages = tf.gather(advantages, idx)

                with tf.GradientTape() as tape:
                    mean = self.policy_net(batch_states)
                    std = tf.exp(self.policy_net.log_std)
                    dist = tfp.Normal(mean, std)
                    log_probs = tf.reduce_sum(dist.log_prob(batch_actions), axis=-1)
                    ratio = tf.exp(log_probs - batch_log_probs_old)
                    surr1 = ratio * batch_advantages
                    surr2 = tf.clip_by_value(ratio, 1-clip_ratio, 1+clip_ratio) * batch_advantages
                    policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                    entropy = tf.reduce_mean(tf.reduce_sum(dist.entropy(), axis=-1))
                    loss = policy_loss - 0.01 * entropy

                grads = tape.gradient(loss, self.policy_net.trainable_variables)
                self.policy_optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))

    def train_value_net(self, states, returns):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        for _ in range(train_value_iters):
            indices = tf.range(states.shape[0])
            indices = tf.random.shuffle(indices)
            for start in range(0, states.shape[0], batch_size):
                end = start + batch_size
                idx = indices[start:end]
                batch_states = tf.gather(states, idx)
                batch_returns = tf.gather(returns, idx)

                with tf.GradientTape() as tape:
                    values = self.value_net(batch_states)
                    value_loss = tf.reduce_mean((values - batch_returns[:, tf.newaxis])**2)
                
                grads = tape.gradient(value_loss, self.value_net.trainable_variables)
                self.value_optimizer.apply_gradients(zip(grads, self.value_net.trainable_variables))

# 训练循环
agent = PPOAgent()

for episode in range(total_episodes):
    states, actions, log_probs_old, rewards, dones = agent.collect_experience(env, max_steps)
    advantages, returns = agent.compute_advantages(rewards, dones, states)
    
    agent.train_policy(states, actions, log_probs_old, advantages)
    agent.train_value_net(states, returns)
    
    total_reward = np.sum(rewards)
    print(f"Episode: {episode}, Total Reward: {total_reward:.1f}")