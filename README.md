# ExO-PPO

面向离散控制与连续控制任务的强化学习实验代码仓库，主要用于研究 PPO、ExO-PPO/GePPO 及其与熵正则、KL 正则、off-policy 采样、SAC、P3O 和 ES 等机制结合的训练方法。

当前仓库以研究和消融实验脚本为主，代码仍在持续整理中。不同脚本可能使用不同的环境版本、超参数和网络结构，运行前请以脚本顶部的配置和注释为准。

## 主要内容

- **Atari**：离散动作空间实验，包含 ExO/PPO、off-policy PPO、SAC/TD、P3O、ES 以及视觉编码器相关实现。
- **MuJoCo/Gymnasium**：连续动作空间实验，包含 PPO、ExO/GePPO、SAC 和基础连续控制示例。
- **向量化采样**：部分脚本使用 `gymnasium.make_vec` 或 EnvPool 并行创建多个环境。
- **训练监控**：部分训练脚本通过 TensorBoard 记录 `score`、`std`、`rate` 等指标。
- **Actor-Critic 网络**：使用 TensorFlow/Keras 实现策略网络和价值网络，并配合 GAE、优势归一化及观测归一化等组件。

## 仓库结构

```text
.
├── Atari/
│   ├── AtariExo*.py       # ExO/PPO 及相关消融实验
│   ├── AtariPPO*.py       # PPO 及 off-policy PPO 变体
│   ├── AtariSACTD*.py     # SAC/TD 相关实验
│   ├── Atariagent*.py     # Atari Actor-Critic 网络
│   ├── Ori-gym/           # 基于原生 Gymnasium/ALE 的示例和对照实验
│   └── vit*.py            # 视觉编码器实验
├── Mujoco/
│   ├── MuExo*.py          # 连续动作 ExO/GePPO 相关实验
│   ├── MuPPO*.py          # 连续动作 PPO 及变体
│   ├── MuSACep.py         # SAC 连续控制实验
│   ├── CtSAC*.py          # 连续 SAC 网络和测试脚本
│   ├── MuagentImEp.py     # MuJoCo Actor-Critic 网络
│   ├── Memo.py            # 经验、观测归一化和动作分布工具
│   ├── ppo_keras_continuous.py
│   │                        # 相对独立的连续动作 PPO 示例
│   └── demo.py/example*.py/tests.py
│                            # 演示、验证和探索性脚本
├── flow/
│   ├── train.py             # direct-ratio ExO-PPO + recent-policy OFP 入口
│   ├── models.py            # interval-average one-step flow actor
│   ├── objectives.py        # ExO ratio 与 OFP 自蒸馏目标
│   ├── torch_train.py       # PyTorch 训练入口（同一方案）
│   ├── torch_models.py      # PyTorch flow actor/value 网络
│   ├── torch_objectives.py  # PyTorch ExO/OFP 目标
│   └── README.md            # 方案公式、运行方式与消融说明
├── .gitignore
└── README.md
```

文件名中的后缀通常对应不同的实验设置或消融项，并不代表统一的命令行参数。需要复现实验时，建议同时记录脚本版本、环境 ID、随机种子和脚本顶部的超参数。

## 环境要求

建议使用独立的 Python 3.x 虚拟环境，并根据本机 CUDA、TensorFlow 和 Keras 版本选择兼容组合。

代码中使用到的主要依赖包括：

- Python、NumPy、SciPy、Numba
- TensorFlow、Keras、TensorFlow Probability
- PyTorch（运行 `flow.torch_train`）
- Gymnasium
- EnvPool（部分 Atari 和并行环境脚本）
- MuJoCo 环境依赖
- Atari/ALE 环境依赖（运行 `Ori-gym` 或 ALE 相关脚本时）

一个基础安装示例：

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install numpy scipy numba tensorflow keras tensorflow-probability gymnasium
python -m pip install envpool
python -m pip install "gymnasium[mujoco]"
```

如果运行 Atari/ALE 脚本，再安装 Atari 相关依赖：

```bash
python -m pip install "gymnasium[atari,accept-rom-license]"
```

如果运行 `LunarLanderContinuous-v2` 等 Box2D 环境，再安装：

```bash
python -m pip install "gymnasium[box2d]"
```

TensorFlow、Keras 和 `tensorflow-probability` 需要相互兼容；如果导入时报版本错误，应优先按照 TensorFlow 版本选择对应的 Keras 和 TensorFlow Probability 版本。EnvPool 在不同操作系统和 Python 版本上的安装支持也可能不同。

## 快速开始

大多数脚本没有统一的 CLI 参数，环境名称、随机种子、并行环境数和训练轮数直接写在脚本顶部。建议从仓库根目录执行脚本：

```bash
# Atari：ExO 变体，默认使用 Breakout-v5
python Atari/AtariExoRe.py

# Atari：off-policy PPO 变体，默认使用 Qbert-v5
python Atari/AtariPPOoff.py

# MuJoCo：连续动作 ExO/GePPO 变体，默认使用 Walker2d-v5
python Mujoco/MuExoPPO.py

# MuJoCo：连续动作 PPO，默认使用 Humanoid-v5
python Mujoco/MuPPO.py

# 相对独立的连续动作 PPO 示例，默认使用 BipedalWalker-v2
python Mujoco/ppo_keras_continuous.py

# MuJoCo：direct-ratio ExO-PPO + recent-policy one-step flow
python -m flow.train --env-id Walker2d-v5

# PyTorch：同一方案，自动选择 CUDA（无 CUDA 时回退 CPU）
python -m flow.torch_train --env-id Walker2d-v5 --device auto

# PyTorch：使用 EnvPool C++ 批量并行采样
python -m flow.torch_train --env-id Walker2d-v5 --env-backend envpool --device auto
```

几个常用脚本及其默认环境如下：

| 脚本 | 默认环境 | 说明 |
| --- | --- | --- |
| `Atari/AtariExoRe.py` | `Breakout-v5` | Atari ExO 变体，使用 EnvPool 并行采样 |
| `Atari/AtariPPOoff.py` | `Qbert-v5` | off-policy PPO 实验 |
| `Atari/AtariSACTDlam.py` | `Pong-v5` | SAC/TD 相关实验 |
| `Mujoco/MuExoPPO.py` | `Walker2d-v5` | 连续动作 ExO/GePPO 实验 |
| `Mujoco/MuPPO.py` | `Humanoid-v5` | 连续动作 PPO 实验 |
| `Mujoco/MuSACep.py` | `LunarLanderContinuous-v2` | 连续控制 SAC 实验 |
| `Mujoco/ppo_keras_continuous.py` | `BipedalWalker-v2` | 相对独立的 PPO 参考实现 |
| `flow/train.py` | `Walker2d-v5` | direct-ratio ExO-PPO + recent-policy one-step flow；详见 `flow/README.md` |
| `flow/torch_train.py` | `Walker2d-v5` | PyTorch direct-ratio ExO-PPO + recent-policy one-step flow |

## GPU 和运行注意事项

许多训练脚本默认使用 GPU，并包含类似下面的配置：

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 部分脚本使用 "1"
gpus = tf.config.experimental.list_physical_devices("GPU")
```

运行前请检查：

1. `CUDA_VISIBLE_DEVICES` 是否指向实际存在的 GPU。
2. 脚本中的显存限制是否适合当前显卡。
3. 如果使用 CPU，需移除或保护 `gpus[0]` 相关配置，否则可能因找不到 GPU 而报错。
4. 如果需要切换 GPU，最好将 `CUDA_VISIBLE_DEVICES` 放到 `import tensorflow` 之前。

CPU 环境可以参考以下保护写法：

```python
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)],
    )
```

Atari 脚本中的环境 ID 并不完全统一，例如同时存在 `Breakout-v5`、`ALE/Breakout-v5`、`Pong-v5`、`Qbert-v5`、`breakout` 以及项目或 EnvPool 特有的环境名。若环境创建失败，请先确认当前依赖实际注册的环境 ID；`BasicMath-v5`、`LeaperEasy-v0` 等名称可能需要对应的自定义环境支持。

## 修改实验配置

通常直接修改目标脚本开头的常量即可：

```python
ENV_NAME = "Walker2d-v5"
GAMMA = 0.99
BATCH = 128
Train_Env_num = 2
Test_Env_num = 5
Run_Step = 256
```

常见配置项包括：

- `ENV_NAME`：Gymnasium、MuJoCo 或 EnvPool 环境名称
- `GAMMA`：折扣因子
- `BATCH`：训练批大小
- `Train_Env_num` / `Test_Env_num`：训练和评估并行环境数量
- `Run_Step`：每次采样的步数
- `EPS`：训练循环规模
- `seed`：随机种子

不同脚本的实现细节和超参数并不完全一致，修改环境或网络结构后应同步检查观测空间、动作空间以及 Actor-Critic 的输入输出维度。

## 日志与模型

部分脚本会将 TensorBoard 日志写入类似下面的目录：

```text
logs/<environment>/<timestamp>-<experiment-name>/
```

启动 TensorBoard 的示例：

```bash
tensorboard --logdir logs
```

部分网络类会使用相对路径保存权重，例如 `disTD/model1/actor`。日志、模型权重、检查点和生成的数据属于实验产物，不建议提交到 Git 仓库；具体保存位置请以对应脚本为准。

## 基础检查

仓库目前没有统一的自动化测试套件。提交改动前，可以先进行 Python 语法检查：

```bash
python -m py_compile Atari/AtariExoRe.py Mujoco/MuExoPPO.py
```

然后用较小的 `Train_Env_num`、`Run_Step` 和 `EPS` 做一次短跑，确认环境、GPU、网络维度和日志路径均正常后，再开始完整训练。

## 当前限制

- 尚未提供统一的 `requirements.txt`、配置文件或命令行入口。
- 部分脚本仍包含硬编码的 GPU 编号和显存限制。
- 不同脚本混用了 Gymnasium、EnvPool 以及不同版本的环境 ID。
- 部分文件属于探索性代码或对照实现，不能保证所有脚本在当前依赖版本下开箱即用。
- 仓库当前未提供独立的实验结果表、预训练模型或许可证文件。

## 贡献建议

新增实验时，建议在脚本开头注明算法变体、环境版本、依赖版本、随机种子、主要超参数和结果保存路径，并尽量避免将日志、模型和大体积数据提交到仓库。
