# Install
```
cd unitree-deplog
uv sync
source .venv/bin/activate
cd unitree_sdk2_python
uv pip install -e .
```

# 启动
真机控制：
`uv run python unitree-deploy/controller.py --mode real --camera --deploy-yaml unitree-deploy/loco_flat/controller.yaml`

仿真控制：
`uv run python unitree-deploy/sim_bridge.py`
`uv run python unitree-deploy/controller.py --mode sim --deploy-yaml unitree-deploy/loco_flat/controller.yaml`

状态可视化：
`uv run python unitree-deploy/visualizer.py --mode sim`

## 控制器状态机
`controller.py` 有 4 个状态：
- `zero_torque_state`
- `move_to_default_qpos`
- `default_qpos_state`
- `run`

切换规则：
- `A` 从零力矩进入默认姿态过渡
- `Start` 从默认姿态进入 `run`
- `X` 返回零力矩

## 仿真按键
`sim_bridge.py` 同时负责遥控器映射和仿真复位：
- `r` 重置仿真到初始状态
- `b` 发送遥控器 `A`
- `m` 发送遥控器 `Start`
- `up` / `down` 调整吊带高度
- `n` 解除吊带
- `w/s/a/d/q/e` 分别控制前后、侧移、转向
- `esc` 退出

## 说明
- `controller.py` 只负责状态订阅和控制输出。
- `sim_bridge.py` 负责仿真状态发布和键盘输入。
- `visualizer.py` 负责纯可视化。

<video controls src="可视化.webm" title="Title"></video>
