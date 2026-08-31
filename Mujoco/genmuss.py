import tensorflow as tf
import numpy as np
import gymnasium as gym
import tensorflow_probability.python.distributions as tfp
import keras

# Hyperparameters
GAMMA = 0.99
CLIP_RATIO = 0.2
LR = 0.0003
EPOCHS = 100
STEPS_PER_EPOCH = 4000
BATCH_SIZE = 64

# Create environment
env = gym.make('Ant-v5')
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# Define the policy network
class PolicyNetwork(keras.Model):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.dense1 = keras.layers.Dense(64, activation='relu')
        self.dense2 = keras.layers.Dense(64, activation='relu')
        self.mu = keras.layers.Dense(act_dim)
        initializer = keras.initializers.Constant(value=-0.5)
        self.log_std = self.add_weight(initializer=initializer, shape=(act_dim,))

    def call(self, obs):
        x = self.dense1(obs)
        x = self.dense2(x)
        mu = self.mu(x)
        std = tf.exp(self.log_std)
        return mu, std

# Define the value network
class ValueNetwork(keras.Model):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.dense1 = keras.layers.Dense(64, activation='relu')
        self.dense2 = keras.layers.Dense(64, activation='relu')
        self.value = keras.layers.Dense(1)

    def call(self, obs):
        x = self.dense1(obs)
        x = self.dense2(x)
        return self.value(x)

policy_net = PolicyNetwork()
value_net = ValueNetwork()
policy_optimizer = keras.optimizers.Adam(learning_rate=LR)
value_optimizer = keras.optimizers.Adam(learning_rate=LR)

def compute_advantages(rewards, values, next_values, dones):
    advantages = np.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * next_values[t] * (1 - dones[t]) - values[t]
        advantages[t] = lastgaelam = delta + GAMMA * 0.95 * (1 - dones[t]) * lastgaelam
    returns = advantages + values
    return advantages, returns

def update_policy(obs, actions, advantages, old_log_probs):
    with tf.GradientTape() as tape:
        mu, std = policy_net(obs)
        dist = tfp.Normal(mu, std)
        log_probs = dist.log_prob(actions)
        ratio = tf.exp(tf.reduce_mean(log_probs - old_log_probs,-1))
        clip_adv = tf.clip_by_value(ratio, 1-CLIP_RATIO, 1+CLIP_RATIO) * advantages
        loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clip_adv))
    grads = tape.gradient(loss, policy_net.trainable_variables)
    policy_optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))

def update_value(obs, returns):
    with tf.GradientTape() as tape:
        values = tf.squeeze(value_net(obs), axis=-1)
        loss = tf.reduce_mean((returns - values) ** 2)
    grads = tape.gradient(loss, value_net.trainable_variables)
    value_optimizer.apply_gradients(zip(grads, value_net.trainable_variables))

def main():
    for epoch in range(EPOCHS):
        obs,_ = env.reset()
        obs_buf, act_buf, adv_buf, ret_buf, logp_buf = [], [], [], [], []
        ep_rews = []
        dones = []
        for t in range(STEPS_PER_EPOCH):
            mu, std = policy_net(tf.expand_dims(obs, axis=0))
            dist = tfp.Normal(mu, std)
            action = dist.sample()[0]
            log_prob = dist.log_prob(action)
            next_obs, reward, done, ter, _ = env.step(action.numpy())
            ep_rews.append(reward)
            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(log_prob)
            dones.append(done)
            obs = next_obs

            if done or ter or (t == STEPS_PER_EPOCH - 1):
                last_val = value_net(tf.expand_dims(obs, axis=0)) if not done else 0
                # rews = np.append(ep_rews, last_val)
                vals = value_net(np.array(obs_buf)).numpy()
                next_vals = np.append(vals[1:], last_val)
                dones_buf = np.array(dones, dtype=np.float32)
                adv_buf, ret_buf = compute_advantages(ep_rews, vals, next_vals, dones)

                obs_buf = np.array(obs_buf, dtype=np.float32)
                act_buf = np.array(act_buf, dtype=np.float32)
                adv_buf = np.array(adv_buf, dtype=np.float32)
                ret_buf = np.array(ret_buf, dtype=np.float32)
                logp_buf = np.array(logp_buf, dtype=np.float32)

                # Update the policy and value networks
                update_policy(obs_buf, act_buf, adv_buf, logp_buf)
                update_value(obs_buf, ret_buf)

                obs_buf, act_buf, adv_buf, ret_buf, logp_buf = [], [], [], [], []
                obs,_ = env.reset()
                ep_rews = []
        print(f'Epoch {epoch + 1}/{EPOCHS} completed')
        evaluate()

def evaluate():
    obs,_ = env.reset()
    reward_n = 0
    for t in range(STEPS_PER_EPOCH):
        mu, std = policy_net(tf.expand_dims(obs, axis=0))
        dist = tfp.Normal(mu, std)
        action = dist.sample()[0]
        obs, reward, done, ter, _ = env.step(action.numpy())
        if done or ter:
            break
        reward_n += reward
    print(reward_n)
    


if __name__ == "__main__":
    main()