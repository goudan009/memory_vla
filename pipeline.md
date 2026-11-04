### 环节1-资源准备（按照需要增加）

###### 1.B1K-2025π0.5的50任务权重

项目来源（IliaLarchenko/behavior_50t_checkpoint）

项目地址：checkpoints/behavior_50t_checkpoint（软链接）

真实地址：/nvme3/projects/SERF-VLA/checkpoints/behavior-1k-solution/behavior_50t_checkpoint

###### 2.B1K-2025任务示教数据

项目地址：datasets/b1k2025/parquet（软链接）

真实地址：/nvme3/projects/SERF-VLA/datasets/2025-BEHAVIOR-1K-CHALLENGE

需要注意：现有38个任务，缺少任务编号为（0001、0007、0008、0010、0015、0019、0022、0025、0027、0032、0036、0044）且现有任务均无图像/videos/

###### 3.B1K-2025仿真资源

**仿真资源入口**

项目地址：datasets/simulation/b1k2025

真实地址：/nvme3/shared/b1k2025/datasets

**场景、物体**

项目地址：datasets/simulation/behavior-1k-assets

真实地址：/nvme3/projects/SERF-VLA/datasets/behavior-1k-assets

**机器人**

项目地址：datasets/simulation/omnigibson-robot-assets

真实地址：/nvme3/projects/SERF-VLA/datasets/omnigibson-robot-assets

**任务实例**

项目地址：datasets/simulation/2025-challenge-task-instances

真实地址：/nvme3/shared/b1k2025/datasets/2025-challenge-task-instances

###### 4.conda运行环境

项目地址：env/b1k2025-sim

真实地址：/nvme3/envs/b1k2025-sim

主要组件：Python 3.10.20、PyTorch 2.5.1+cu121、Isaac Sim 4.5.0、OmniGibson 3.7.1

###### 5.参考文章

项目地址：/nvme3/projects/memory_vla/paper/references(我已存好)

###### 6.参考代码

项目地址：/nvme3/projects/memory_vla/src/references（只放代码，不放大文件，开源了的项目：serf、memoryvla、mem）

###### 7.仿真引擎

**Isaac Sim 4.5**

项目地址：env/isaac-sim-4.5

真实地址：/nvme3/envs/b1k2025-sim/lib/python3.10/site-packages/isaacsim

**OmniGibson 源码**

项目地址：src/simulation/omnigibson

真实地址：/nvme3/shared/b1k2025/BEHAVIOR-1K/OmniGibson

**JOYLO 任务配置及辅助代码**

项目地址：src/simulation/joylo

真实地址：/nvme3/shared/b1k2025/BEHAVIOR-1K/joylo

###### 8.资源清单/项目架构构建

放跟本项目相关的资源清单

项目地址：

### 环节2-资源测试

###### 1.
