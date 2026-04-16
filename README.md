# Install
```
cd unitree-deplog
uv sync
source .venv/bin/activate
cd unitree_sdk2_python
uv pip install -e .
```

# Real
`uv run python unitree-deploy/g1_low_level_example.py --mode real --camera`
# Sim
```
uv run python unitree-deploy/sim_bridge.py

uv run python unitree-deploy/deploy.py

uv run python unitree-deploy/g1_low_level_example.py
```
## Sim 按键说明
程序或重置后在零力矩模式，依次按下以下按键
- `r`恢复初始仿真状态，即零力矩模式
- `b`进入阻尼模式，即默认位置（双手端平）
- `up`和`down`调整吊带位置
- `n`解除龙门架的带子
- `m`开始接收外部指令


相机序列号 `rs-enumerate-devices`

存在问题：
1. 重置后机器人乱飞
2. 需要在终端输入键盘指令

<video controls src="可视化.webm" title="Title"></video>