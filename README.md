REALSENSE_ENABLED=1 uv run python unitree-deploy/g1_low_level_example.py

相机序列号 `rs-enumerate-devices`

存在问题：
1. 相机要插拔
[WARN] Failed to create RealSense camera d435_head: xioctl(VIDIOC_S_FMT) failed, errno=16 Last Error: Device or resource busy
2. viser文件中计算forward
3. viser文件不需要main

<video controls src="可视化.webm" title="Title"></video>