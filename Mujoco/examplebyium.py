import tensorflow as tf
import tensorflow_probability.python.distributions as tfp
from keras import losses, layers,Model
from keras import optimizers
import keras
import gymnasium as gym
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
ENV_NAME = 'Ant-v5'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
# 超参数设置
GAMMA = 0.99
CLIP_EPSILON = 0.2
CRITIC_DISCOUNT = 0.5
ENTROPY_BONUS = 0.01
LEARNING_RATE = 3e-4
EPOCHS = 10
BATCH_SIZE = 64
TRAIN_STEPS = 2048000
UPDATE_STEPS = 10

# 环境初始化
env = gym.make('Ant-v5')
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# 策略网络
class Actor(Model):
    def __init__(self, action_dim):
        super(Actor, self).__init__()
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.mean = layers.Dense(action_dim)
        initializer = keras.initializers.Constant(value=-1.6)
        self.log_std = self.add_variable(initializer=initializer, shape=(action_dim,), trainable=True)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        mean = self.mean(x)
        std = tf.exp(self.log_std)
        return mean, std

# 价值网络
class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.value = layers.Dense(1)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        value = self.value(x)
        return value

# PPO算法
class PPO:
    def __init__(self, actor: Actor, critic:Critic):
        self.actor:Actor = actor
        self.critic:Critic = critic
        self.optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)

    def select_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        mean, std = self.actor(state)
        dist = tfp.Normal(mean, std)
        action = tf.squeeze(dist.sample(), axis=0)
        log_prob = tf.reduce_sum(dist.log_prob(action), axis=-1)
        return action.numpy(), log_prob.numpy()

    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + GAMMA * 0.95 * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return np.array(advantages)

    def train(self, states, actions, log_probs, returns, advantages):
        for _ in range(EPOCHS):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            for start in range(0, len(states), BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_indices = indices[start:end]
                with tf.GradientTape() as tape:
                    mean, std = self.actor(tf.convert_to_tensor(states[batch_indices], dtype=tf.float32))
                    dist = tfp.Normal(mean, std)
                    new_log_probs = tf.reduce_sum(dist.log_prob(actions[batch_indices]), axis=-1)
                    entropy = tf.reduce_sum(dist.entropy(), axis=-1)
                    ratios = tf.exp(new_log_probs - log_probs[batch_indices])
                    advantages_batch = tf.convert_to_tensor(advantages[batch_indices], dtype=tf.float32)
                    surr1 = ratios * advantages_batch
                    surr2 = tf.clip_by_value(ratios, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * advantages_batch
                    actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2) + ENTROPY_BONUS * entropy)
                    values = self.critic(tf.convert_to_tensor(states[batch_indices], dtype=tf.float32))
                    critic_loss = CRITIC_DISCOUNT * tf.reduce_mean(tf.square(returns[batch_indices] - tf.squeeze(values)))
                    loss = actor_loss + critic_loss
                grads = tape.gradient(loss, self.actor.trainable_variables + self.critic.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables + self.critic.trainable_variables))

# 训练循环
def train_ppo(env:gym.Env, ppo:PPO, train_steps, update_steps):
    all_episode_rewards = []
    state = env.reset()[0]
    episode_reward = 0
    states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

    for step in range(1, train_steps + 1):
        action, log_prob = ppo.select_action(state)
        next_state, reward, done, ter,  _ = env.step(action)
        value = ppo.critic(tf.convert_to_tensor([state], dtype=tf.float32))
        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        dones.append(done)
        values.append(value.numpy()[0][0])
        episode_reward += reward
        
        state = next_state

        if done or ter:
            next_state = env.reset()[0]
            all_episode_rewards.append(episode_reward)
            episode_reward = 0

        if step % update_steps == 0:
            next_value = ppo.critic(tf.convert_to_tensor([next_state], dtype=tf.float32)).numpy()[0][0]
            values.append(next_value)
            returns = []
            discounted_sum = next_value
            for reward, done in zip(reversed(rewards), reversed(dones)):
                discounted_sum = reward + GAMMA * discounted_sum * (1 - done)
                returns.insert(0, discounted_sum)
            returns = np.array(returns)

            advantages = ppo.compute_advantages(rewards, values[:-1], values[1:], dones)
            ppo.train(np.array(states), np.array(actions), np.array(log_probs), returns, advantages)

            states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

        if step % 1000 == 0:
            print(f"Step: {step}, Avg. Reward: {np.mean(all_episode_rewards[-10:])}")

    return all_episode_rewards

# 初始化Actor和Critic网络
actor = Actor(action_dim)
critic = Critic()
ppo = PPO(actor, critic)

# 开始训练
train_rewards = train_ppo(env, ppo, TRAIN_STEPS, UPDATE_STEPS)

# 训练结束，输出结果
print("Training finished.")
print(f"Average reward over last 10 episodes: {np.mean(train_rewards[-10:])}")