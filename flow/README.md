# Direct-ratio ExO-PPO + recent-policy One-Step Flow

这是一套面向连续动作空间的实验方案。它保留 ExO-PPO 可直接计算的
importance ratio，同时把策略均值换成参考 OFP（One-Step Flow Policy）的
interval-average flow。实现入口是：

```bash
python -m flow.train --env-id Walker2d-v5
```

PyTorch 版本使用同一套增广策略和目标函数，入口为：

```bash
python -m flow.torch_train --env-id Walker2d-v5 --device auto
```

## 1. 为什么要使用增广策略

普通 flow policy 的边缘密度 `pi(a | s)` 通常不可解，直接把它塞进 PPO 的
`pi_new / pi_behavior` 会得到一个并不存在的“精确 ratio”。本实现显式保留
flow latent：

```text
z ~ Normal(0, I)
mu_theta(s, z, t) = z + (1 - t) u_theta(z, t, 1 | s)
x ~ Normal(mu_theta, diag(sigma_theta^2))
a_env = action_center + action_scale * tanh(x)
```

把 `(z, x, t)` 与 rollout 一起保存后，增广策略的 ratio 为：

```text
r(theta) = q_theta(z, x | s) / q_behavior(z, x | s)
         = Normal(x; mu_theta(s,z,t), sigma_theta)
           / Normal(x; mu_behavior(s,z,t), sigma_behavior)
```

`p(z)` 在分子和分母中抵消；tanh 与动作缩放的 Jacobian 也抵消。因此这是
条件增广空间上的 direct ratio，不需要 flow likelihood、ODE divergence 或
额外 ratio estimator。边缘动作分布仍然是对 `z` 积分得到的非高斯混合分布。

## 2. recent-policy ExO 中心

每轮采样结束、开始 minibatch 更新前冻结 `pi_recent`。对于 replay 中由
`pi_behavior` 生成的样本，分别计算：

```text
r_direct = pi_theta(x | s,z) / pi_behavior(x | s,z)
r_recent = pi_recent(x | s,z) / pi_behavior(x | s,z)
```

ExO 的平滑区间以逐样本的 `r_recent` 为中心，而不是永远以 1 为中心：

```text
[r_recent - epsilon, r_recent + epsilon]
```

因此最新 rollout 上 `r_recent` 约为 1；较旧 replay 上的中心会自动反映
recent policy 相对旧 behavior policy 的位移。actor 主目标为：

```text
L_exo = -E[min(r_direct * A, exo(r_direct; r_recent) * A)]
```

`exo()` 在区间内是恒等映射，在区间外使用 ExO-PPO 的指数软尾部。

## 3. recent-policy one-step flow

论文中的 OFP 用 expert action 作为 endpoint；在线 RL 没有 expert dataset，
这里改为从冻结的 recent policy 生成 endpoint，并复用其已知 latent 形成配对
传输：

```text
z ~ Normal(0, I)
x_recent = z + u_recent(z, 0, 1 | s)
```

注意这里只蒸馏 recent flow endpoint，不重复蒸馏用于探索的 residual Gaussian，
否则每次 policy sync 都会叠加一次探索方差。

实现包含 OFP 的三个目标：

- boundary flow anchoring：在 `t == r` 上回归瞬时速度；
- EMA self-consistency：使用嵌套区间和 time-contracting schedule 构造只需
  forward pass 的 teacher target；
- self-guidance：利用 EMA 的 conditional/unconditional gap 提供 CFG 方向。

联合 actor 损失为：

```text
L_actor = L_exo
        + lambda_ofp * (
              0.8 * L_flow
            + 0.2 * L_consistency
            + 0.05 * L_guidance
          )
        - entropy_coefficient * H[x | s,z]
```

权重均可通过 CLI 调整。`pi_recent` 是一个整轮不变的 proximal snapshot；
`pi_ema` 是每个 minibatch 后更新的自蒸馏 teacher，两者不能混用。

## 4. 一次训练迭代

1. 当前策略以一次 flow forward 产生动作，同时记录 `z`、pre-tanh action 和
   behavior log-probability。
2. 用 GAE 计算 rollout 的 advantage/return，并放入有界 replay window。
3. 冻结当前网络为 `pi_recent`。
4. replay minibatch 上计算 direct ratio 与 recent ratio，更新 ExO actor 和
   critic。
5. 同一批 state 上以 `pi_recent` endpoint 计算 OFP 三项损失。
6. 更新在线策略后，再更新 EMA teacher。

这套 replay 仍与原仓库的 ExO/GePPO 一样只修正动作分布，不修正旧 replay 的
state-distribution shift；`replay_rollouts` 不宜设得很大。

## 5. 运行

先按根目录 README 安装对应的 PyTorch/TensorFlow、Gymnasium 和 MuJoCo 依赖。当前实现已用
Python 3.10、TensorFlow CPU 2.15.1、Gymnasium 1.1.1 完成目标函数数值测试，
用 `Pendulum-v1` 完成训练/评估冒烟，并用 `Walker2d-v5` 完成 64 环境步的
采样、双 rollout replay 和训练冒烟。运行 MuJoCo 仍需额外安装
`gymnasium[mujoco]`。

最小短跑：

```bash
python -m flow.train \
  --env-id Walker2d-v5 \
  --total-steps 2048 \
  --num-envs 2 \
  --rollout-steps 64 \
  --replay-rollouts 2 \
  --warmup-rollouts 2 \
  --update-epochs 1 \
  --batch-size 64 \
  --eval-every-rollouts 999
```

PyTorch 短跑（`--device cuda` 可显式指定 GPU）：

```bash
python -m flow.torch_train \
  --env-id Walker2d-v5 \
  --device auto \
  --total-steps 2048 \
  --num-envs 2 \
  --rollout-steps 64 \
  --replay-rollouts 2 \
  --warmup-rollouts 2 \
  --update-epochs 1 \
  --batch-size 64 \
  --eval-every-rollouts 999
```

如果已安装 EnvPool，可以将采样后端切换为 C++ 批量环境池。这里固定
`batch_size=num_envs` 使用同步批量 API，保持 GAE 和 replay 的环境行顺序：

```bash
python -m flow.torch_train \
  --env-id Walker2d-v5 \
  --env-backend envpool \
  --envpool-num-threads 8 \
  --device auto
```

EnvPool 需要单独安装（`pip install envpool`），并且环境 ID 必须出现在
`envpool.list_all_envs()` 中；不支持的任务会在启动时给出明确错误。异步
`send/recv` 暂未接入，因为它返回乱序的 `env_id`，需要改变当前固定批次的
replay/GAE 数据布局。

正式实验示例：

```bash
python -m flow.train \
  --env-id Walker2d-v5 \
  --total-steps 1000000 \
  --num-envs 4 \
  --rollout-steps 256 \
  --replay-rollouts 4 \
  --warmup-rollouts 4 \
  --update-epochs 2 \
  --batch-size 256 \
  --exo-beta 5 \
  --exo-clip-radius 0.2 \
  --ofp-coef 0.1
```

日志默认写入 `logs/<env>-direct-ratio-recent-ofp-<timestamp>/`：

```bash
tensorboard --logdir logs
```

### 建议消融

```bash
# A. 关闭 OFP，只保留 direct-ratio recent-centered ExO
--ofp-coef 0

# B. 关闭 self-guidance
--guidance-mix 0

# C. 退化为只使用最新 rollout
--replay-rollouts 1 --warmup-rollouts 1

# D. 单动作版 temporal warm start（默认关闭）
--warm-start-time 0.15
```

论文的 warm-start 针对 action chunk。这里提供的单动作版本用上一时刻的
pre-tanh action 构造 `z_t = (1-t) epsilon + t a_previous`；它是实验开关，
不应与论文的 action-chunk 结果直接等同。

## 6. 实现边界

- 当前入口支持 flat `Box` observation 和 flat、有限边界的 `Box` action。
- OFP 论文是 imitation learning；这里的 recent-policy endpoint 是在线 RL
  适配，不是论文原实验的逐项复现。
- 默认确定性评估使用 `z=0`。加 `--stochastic-eval` 可评估完整混合策略。
- ratio 必须使用 replay 中保存的 pre-tanh action 和 flow latent；只保存环境
  action 后再做 `atanh` 会在饱和区产生明显数值误差。
- PyTorch 实现对应的文件是 `torch_models.py`、`torch_objectives.py` 和
  `torch_train.py`；检查点使用 `torch.save` 保存为 `.pt`，内容包含 policy、
  value、观测统计量和 `update_step`。

## 参考

- [One-Step Flow Policy: Self-Distillation for Fast Visuomotor Policies](https://arxiv.org/abs/2603.12480v1)
- [ExO-PPO repository baseline](../Mujoco/MuExoPPO.py)
