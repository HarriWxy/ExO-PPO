import gymnasium as gym
import tensorflow as tf
from keras import layers, Model
import numpy as np
import keras

# 创建环境
env = gym.make('Ant-v5')
num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

# PPO超参数
clip_ratio = 0.2
lr = 0.0003
epochs = 1000
gamma = 0.99
lam = 0.95

# 创建actor模型
def get_actor():
    inputs = layers.Input(shape=(num_states,))
    common = layers.Dense(256, activation="relu")(inputs)
    common = layers.Dense(256, activation="relu")(common)
    outputs = layers.Dense(num_actions, activation="tanh")(common)
    outputs = layers.Lambda(lambda x: x * upper_bound)(outputs)
    return keras.Model(inputs, outputs)

# 创建critic模型
def get_critic():
    inputs = layers.Input(shape=(num_states,))
    common = layers.Dense(256, activation="relu")(inputs)
    common = layers.Dense(256, activation="relu")(common)
    outputs = layers.Dense(1)(common)
    return keras.Model(inputs, outputs)

# 计算优势函数
def get_advantages(values, masks, rewards, gamma, lam):
    advs = np.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * masks[t] - values[t]
        advs[t] = lastgaelam = delta + gamma * lam * masks[t] * lastgaelam
    returns = advs + values[:-1]
    return returns, advs

# 更新模型参数
@tf.function
def update(actor:Model, critic:Model, states, actions, advantages, returns):
    with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
        pred = actor(states, training=True)
        values = critic(states, training=True)
        values = tf.squeeze(values)

        critic_loss = keras.losses.mean_squared_error(returns, values)

        ratios = tf.exp(tf.reduce_sum(tf.square(actions - pred), axis=1))
        surr1 = ratios * advantages
        surr2 = tf.clip_by_value(ratios, 1 - clip_ratio, 1 + clip_ratio) * advantages
        actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

    grads1 = tape1.gradient(actor_loss, actor.trainable_variables)
    grads2 = tape2.gradient(critic_loss, critic.trainable_variables)
    actor.optimizer.apply_gradients(zip(grads1, actor.trainable_variables))
    critic.optimizer.apply_gradients(zip(grads2, critic.trainable_variables))

actor = get_actor()
critic = get_critic()
actor.compile(optimizer=keras.optimizers.Adam(lr))
critic.compile(optimizer=keras.optimizers.Adam(lr))

# 训练PPO算法
for epoch in range(epochs):
    states = env.reset()[0]
    rewards = []
    actions = []
    states_memory = []
    masks = []
    values = []

    for t in range(2000):
        # env.render()
        state = tf.expand_dims(states, 0)
        action = actor(state)
        action = action.numpy()[0]
        next_state, reward, done, ter, _ = env.step(action)
        
        actions.append(action)
        rewards.append(reward)
        states_memory.append(states)
        masks.append(1 - done)
        values.append(critic(state).numpy())

        states = next_state
        if done or ter:
            break

    values = np.array(values).flatten()
    rewards = np.array(rewards)
    masks = np.array(masks)
    states_memory = np.array(states_memory)
    actions = np.array(actions)
    values = np.append(values, critic(tf.expand_dims(next_state, 0)).numpy())

    returns, advantages = get_advantages(values, masks, rewards, gamma, lam)
    advantages =  tf.convert_to_tensor(advantages, dtype=tf.float32)

    update(actor, critic, states_memory, actions, advantages, returns)

    print(f"Epoch {epoch + 1}/{epochs}, Total Reward: {np.sum(rewards)}")

env.close()
