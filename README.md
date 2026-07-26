# Darwinian Memory System 复现实验

## 1. 项目介绍

本项目在 AndroidWorld 动态 GUI 环境中复现 Darwinian Memory System（DMS），并统一比较三种方法：

- Baseline A：不使用跨任务记忆的 PA-Lite Planner–Actor。
- Baseline B：按时间顺序追加历史轨迹的静态记忆。
- DMS：支持分层存储、双因子检索、轨迹回放、风险反馈、变异替换和动态剪枝的记忆系统。

三种方法使用相同的 AndroidWorld evaluator、任务顺序、随机种子、动作预算和
Qwen2.5-VL-7B-Instruct 模型。实验重点是观察跨轮记忆能否提高任务成功率，同时降低
单任务 Token 和 Step 用量。

当前已经完成的正式实验采用 5 个固定任务、5 轮、3 种方法，共 75 次计分运行。该运行通过
了 a11y、ADB、forwarder、Chrome native crash 和保存 observation 的基础设施审计。

## 2. 仓库结构

```text
DMS/
├── README.md                 # 项目说明、运行方法、结果和限制
├── pytest.ini                # 默认只收集本项目 tests/，排除 vendored AndroidWorld 测试
├── configs/                  # 三种方法、远程模型与 AndroidWorld 运行配置
├── datasets/                 # AndroidWorld 数据集与正式五任务清单
├── docs/                     # 机制对应、环境和实验协议说明
├── figs/                     # 经过确认的正式图表、CSV 和汇总
├── runs/                     # 每次运行的轨迹、observation、日志和逐任务结果
├── scripts/
│   ├── common/               # 所有脚本共用的环境激活逻辑
│   ├── setup/                # Python、AndroidWorld、模型与运行资源安装
│   ├── run/                  # 单方法、三方法及模拟器启动入口
│   ├── analysis/             # 结果汇总与绘图
│   ├── monitor/              # AndroidWorld、a11y 与模型连通性检查
│   └── data/                 # 数据集生成与校验
├── src/
│   ├── dms/                  # PA-Lite、静态记忆、DMS 与 runner
│   ├── env/                  # a11y、ADB、observation 和环境适配
│   └── model_client/         # 本地及远程 Qwen-VL 客户端
├── third_party/
│   └── android_world/        # 固定版本的上游 AndroidWorld 源码
├── tests/                    # 单元测试与集成测试
└── requirements.txt          # Python 3.10 依赖版本
```

`runs/<run_id>/figs/` 保存某一次实验自动生成的图；仓库根目录的 `figs/` 只保存经过审计、
需要展示或写入报告的正式结果。`runs/`、模型权重、conda 环境、Android SDK、AVD 和日志等
大型运行产物不提交到 Git。

正式 Python 包名为 `dms`，上游 AndroidWorld 独立存放在 `third_party/android_world/`。
目录重构只改变源码组织和启动路径，没有修改算法、Prompt、实验配置或已生成结果。

## 3. 实验环境

本地主机与 WSL：

- Windows 10 Pro 64-bit 宿主机。
- WSL2，Ubuntu 24.04.3 LTS。
- Python 3.10.18，项目环境位于 `conda_envs/dms_py310`。
- AMD Ryzen 5 7500F，6 核 12 线程，约 31.7 GiB RAM。
- Pixel 6 / Android API 33 / x86_64 模拟器。
- AVD：`AndroidWorldAvd`。
- ADB device：`emulator-5554`。
- AndroidWorld gRPC：8554。
- 模拟器以 headless 模式运行，启用 `-feature -Vulkan`，并使用 SwiftShader/llvmpipe
  软件图形链路规避 Chrome GPU compositor 的 native crash。
- 主 observation 来源是 accessibility forwarder 与 screenshot；正式运行采用严格 a11y
  协议，实际 UIAutomator 回退或仅包含 SystemUI 的 observation 均视为基础设施污染。

远程推理端：

- Ubuntu 22.04.1 LTS。
- NVIDIA RTX 4090，24564 MiB。
- vLLM 0.10.2。
- `Qwen/Qwen2.5-VL-7B-Instruct`，bfloat16。
- `max_model_len=32768`，`max_num_seqs=1`。
- `gpu_memory_utilization=0.92`。
- OpenAI-compatible API 通过 SSH 本地转发暴露为
  `http://127.0.0.1:8000/v1`。
- `do_sample=false`，单次最大输出 192 tokens。

Android emulator、ADB、accessibility forwarder 和 evaluator 全部由 WSL 工作区控制；截图和
a11y tree 通过本地 SSH 转发发送给远程模型，模型返回结构化动作后再由 WSL 中的 AndroidWorld
执行。远程服务只负责模型推理，不参与 memory 存储或 evaluator 判定。

首次部署可在 WSL 项目根目录执行：

```bash
bash scripts/setup/bootstrap_clone_setup.sh
```

只需激活现有环境时执行：

```bash
source scripts/common/activate_env.sh
```

## 4. 一键运行五任务实验

运行前需要确保：

1. AndroidWorld 模拟器已经启动并完成 boot。
2. accessibility forwarder 正常运行。
3. SSH 隧道已经将远程 vLLM 映射到 WSL 的 `127.0.0.1:8000`。
4. `GET http://127.0.0.1:8000/v1/models` 可以正常返回。

在 WSL 项目根目录执行：

```bash
cd /path/to/DMS
bash scripts/run/run_selected5_all_methods.sh
```

脚本默认使用 `datasets/mini_benchmark_probe5.yaml`，按顺序运行 Baseline A、Baseline B 和
DMS，每种方法执行 1 轮。运行前会验证数据集恰好包含 5 个任务，并检查模型服务、
AndroidWorld 和 a11y。

运行正式 5 轮实验：

```bash
bash scripts/run/run_selected5_all_methods.sh --rounds 5
```

指定另一份五任务数据集：

```bash
bash scripts/run/run_selected5_all_methods.sh \
  --rounds 5 \
  --dataset datasets/my_selected_5tasks.yaml
```

只验证输入并打印将要执行的命令，不启动实验：

```bash
bash scripts/run/run_selected5_all_methods.sh --rounds 5 --dry-run
```

每次执行都会创建全新目录：

```text
runs/selected5_3methods_<rounds>rounds_<timestamp>/
├── launcher.stdout.log
├── current_method.txt
├── baseline_a.stdout.log
├── baseline_b.stdout.log
├── dms.stdout.log
├── baseline_a/
├── baseline_b/
└── dms/
```

终端会实时打印每个 step 的动作与结果，以及每个任务的 success、steps、tokens 和
memory size。完整 observation、原始 JSONL、单任务轨迹、metrics 和 memory 审计仍保存在
对应方法目录中。脚本拒绝覆盖或自动接续非空 RunRoot。

## 5. 三种对比方法

| 方法 | 机制 | 主要文件 |
| --- | --- | --- |
| Baseline A | PA-Lite Planner–Actor；每个任务不读取跨任务记忆 | `src/dms/agent.py`、`src/dms/prompts.py` |
| Baseline B | 按时间顺序追加完整历史；不做检索、反馈调节或剪枝 | `src/dms/static_memory.py`、`src/dms/runner.py` |
| DMS | 分层记忆、双因子检索、风险抑制、轨迹回放、变异替换和容量剪枝 | `src/dms/darwinian_memory.py`、`src/dms/agent.py` |

三种方法由同一个 runner 启动，正式入口为 `python -m dms.runner`。DMS memory 使用
`all-MiniLM-L6-v2` 生成检索 embedding；Qwen2.5-VL-7B-Instruct 负责 Planner 和 Actor
决策。

DMS 当前配置的初始容量为 24，最大容量为 96。动态剪枝只在 active memory 数量达到当前
容量时计算 survival-value elbow；`failure_count` 影响 survival/risk 评分，而连续 replay
验证失败达到 `verification_limit=3` 才触发风险删除。

## 6. 实验结果

### 6.1 基础设施诊断与正式运行条件

早期实验曾出现 accessibility forwarder 崩溃、空 a11y tree 转 UIAutomator、ADB dump
失败，以及模型把 forwarder 崩溃页面学习进 memory 的情况。这些运行只能证明 runner、
远程模型、模拟器、evaluator 和 memory 存储链路能够执行，不能用于判断三种算法能力。

当前 WSL 运行链路完成了以下稳定化：

1. a11y 获取失败在严格协议下作为基础设施错误上抛，不再静默保存错误 observation。
2. 模拟器启用 Vulkan feature 与软件图形后端，规避已观察到的 Chrome GPU native crash。
3. Chrome 首次启动页面、DocumentsUI/Files 映射和 Downloads 导航由短路逻辑处理。
4. `tap(index, expected_text)` 在 index 与文字不一致时只接受唯一可见文字匹配，避免索引漂移误点。
5. 只有 AndroidWorld `InformationRetrieval` 任务允许把 `complete` 理由转换成答案，避免
   `BrowserDraw` 因关键词误判进入 answer 循环。
6. 每个正式实验使用新的 RunRoot，不拼接或续接受到污染的旧结果。

最新 75 次正式运行没有记录 runtime error；日志审计未发现实际 UIAutomator 回退、
新 forwarder 崩溃、Chrome native crash 或被保存的 systemui-only observation。

### 6.2 五任务五轮正式实验

正式任务固定为：

| 任务 | Seed |
| --- | ---: |
| `AudioRecorderRecordAudio` | 1030 |
| `RecipeAddSingleRecipe` | 1031 |
| `CameraTakePhoto` | 1032 |
| `BrowserDraw` | 1033 |
| `ClockStopWatchRunning` | 1034 |

结果来自：

```text
runs/main_3methods_5tasks_5rounds_vulkan_20260726_004730
```

严格成功要求 AndroidWorld evaluator 判定成功，并在该任务官方
`int(10 × complexity)` action budget 内完成。

| 方法 | 成功数 | 成功率 | 平均 Token/任务 | 平均 Step/任务 | Runtime Errors | 最终 Memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline A | 5/25 | 20.00% | 67,594.0 | 13.88 | 0 | 0 |
| Baseline B | 9/25 | 36.00% | 80,411.1 | 13.44 | 0 | 25 |
| DMS | 13/25 | 52.00% | 42,119.9 | 12.20 | 0 | 12 |

DMS 在这个固定五任务 mini benchmark 上成功率最高。相较 Baseline A，DMS 平均单任务
Token 少约 37.7%，Step 少约 12.1%；相较 Baseline B，Token 少约 47.6%，Step 少约
9.2%。

逐轮成功数：

| 方法 | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline A | 1/5 | 1/5 | 1/5 | 1/5 | 1/5 |
| Baseline B | 1/5 | 2/5 | 1/5 | 2/5 | 3/5 |
| DMS | 2/5 | 2/5 | 3/5 | 3/5 | 3/5 |

逐任务成功数：

| 任务 | Baseline A | Baseline B | DMS |
| --- | ---: | ---: | ---: |
| `AudioRecorderRecordAudio` | 0/5 | 3/5 | 3/5 |
| `RecipeAddSingleRecipe` | 0/5 | 0/5 | 0/5 |
| `CameraTakePhoto` | 5/5 | 5/5 | 5/5 |
| `BrowserDraw` | 0/5 | 0/5 | 0/5 |
| `ClockStopWatchRunning` | 0/5 | 1/5 | 5/5 |

DMS 的逐轮 memory size 为 `6 → 6 → 9 → 10 → 12`。由于 active memory 从未达到
`min_capacity=24`，本次运行没有触发容量剪枝；这不能解释为剪枝失效。要观察与容量相关的
增长—下降过程，需要扩大任务数量，或者使用单独标记为非正式的低容量诊断配置。

#### 1. 三种算法的成功率随轮数变化

![Success rate by round](figs/success_rate_by_round.png)

#### 2. 三种算法的平均单任务 Token 用量随轮数变化

![Average tokens per task by round](figs/avg_tokens_per_task_by_round.png)

#### 3. 三种算法的平均单任务 Step 数随轮数变化

![Average steps per task by round](figs/avg_steps_per_task_by_round.png)

#### 4. DMS 记忆库大小随任务时间变化

横轴中的一个时间单位对应一个已经完成的 DMS 任务尝试；每 5 个时间单位为一轮。

![DMS memory size timeline](figs/dms_memory_size_timeline.png)

#### 5. 五个单独任务的逐轮成功/失败

![AudioRecorderRecordAudio success by round](figs/task_success_by_round/AudioRecorderRecordAudio_success_by_round.png)
![RecipeAddSingleRecipe success by round](figs/task_success_by_round/RecipeAddSingleRecipe_success_by_round.png)
![CameraTakePhoto success by round](figs/task_success_by_round/CameraTakePhoto_success_by_round.png)
![BrowserDraw success by round](figs/task_success_by_round/BrowserDraw_success_by_round.png)
![ClockStopWatchRunning success by round](figs/task_success_by_round/ClockStopWatchRunning_success_by_round.png)

完整逐轮数值、逐任务 CSV 和基础设施错误审计位于 `figs/summary.md`、
`figs/round_metrics.csv`、`figs/task_results.csv` 和 `figs/task_error_audit.json`。

## 7. Gap 分析：7B 与论文 72B

论文实验使用的 72B 视觉语言模型，在长程规划、细粒度定位、结构化动作生成和错误恢复上
明显强于当前复现使用的 7B 模型。7B 更容易误点相邻控件、输出不完整工具调用，或在任务状态
已经变化后继续执行旧计划。

当前实现保留统一的 Planner–Actor 主骨架和核心 Prompt，并尽量把平台适配限制在运行层：
a11y 生命周期、Chrome/Files onboarding、动作目标校验、SSH 本地端口转发和远程
OpenAI-compatible 客户端。这些适配对三种方法共同生效，不为 DMS 单独提供任务答案。

当前结果仍有以下限制：

- 只覆盖 5 个固定任务和固定 seed，样本量较小。
- `RecipeAddSingleRecipe` 和 `BrowserDraw` 三种方法均未成功，任务覆盖不均衡。
- DMS memory 没有达到容量阈值，因此本次结果没有验证动态剪枝的端到端效果。
- 远程模型、模拟器启动时序和 GUI 状态可能造成运行间波动，不能保证逐动作完全一致。
- 7B mini benchmark 的成功率不能直接等同于论文 72B 在完整 AndroidWorld 上的结果。

因此，这次实验支持“在当前五任务设置下，DMS 的成功率和资源效率优于两个 baseline”，但
不能外推为 DMS 在完整 AndroidWorld、其他模型规模或所有任务类型上都具有同样优势。
