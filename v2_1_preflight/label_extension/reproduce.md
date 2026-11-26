# 复现与扩展

在仓库根目录使用`/nvme3/envs/b1k2025-sim/bin/python`。以下只进行CPU数据处理，不启动Isaac、训练或模型推理。先在工作副本保留已有产物；依赖上级调查目录的完整decoded NPZ、原始200条HDF5、82维parquet和原分辨率三视角视频，路径见上级报告。

```sh
/nvme3/envs/b1k2025-sim/bin/python v2_1_preflight/scripts/recover_task40_assets.py
/nvme3/envs/b1k2025-sim/bin/python v2_1_preflight/scripts/prepare_task40_geometry.py
/nvme3/envs/b1k2025-sim/bin/python v2_1_preflight/scripts/build_task40_extension.py
/nvme3/envs/b1k2025-sim/bin/python v2_1_preflight/scripts/finalize_task40_extension.py
/nvme3/envs/b1k2025-sim/bin/python v2_1_preflight/scripts/validate_inside_source.py
/nvme3/envs/b1k2025-sim/bin/python v2_1_preflight/scripts/audit_task40_extension.py
/nvme3/envs/b1k2025-sim/bin/python v2_1_preflight/scripts/package_extension.py
```

恢复脚本需要官方网络访问和requests；几何脚本需要既有本机授权密钥、cryptography和pxr，但不输出密钥/解密美术。标签构建需numpy/scipy/pandas/pyarrow/h5py，图像审核需av/Pillow，源码适配对照需CPU torch及所记录版本的OmniGibson源码。源码副本在本目录`source_evidence/`；实际复现脚本默认读取已记录的本机源码路径，迁移机器时需替换该显式路径。

仅用标签时不需要仿真和资产：

```python
import pandas as pd
df = pd.read_parquet('v2_1_preflight/label_extension/task40_labels.parquet')
strict = df['door_ever_opened_certified_valid']
weak_inside = df['bag_inside_weak_valid']
# 选择何种监督证据属于后续模型方案；本调查没有自动将weak纳入训练。
```

其他任务直接import `scripts/label_api.py`的接口，给出按观测t严格对齐的pose、真实几何和物理单位，不按Task40时间百分比或动作名称转换阶段。`ever(Label(...))`每个episode单独调用。输出`Label.columns(name)`统一生成值、quality及掩码。

Inside数值对照只调用原源码方法体和轻量适配对象；恢复几何尚未认证原始身份，调用结果一致不会自动改变quality。添加其他任务的真实仿真校验时遵循仓库工作台preview→资源核对→start流程，不能将本CPU脚本称为已做实时仿真验证。
