# 模型版本、测试/实验编号、训练次数与 checkpoint 命名

## 唯一、稳定、可追溯的五段名称

规范：`v{model_version}__test{test_index}__exp{experiment_number}__run{training_run}__step{optimizer_step}`。

| 字段 | 示例 | 含义 | 是否因续训改变 |
|---|---|---|---|
| `v1` | `v1` | V1模型方法版本；V2须由用户确认方法变更 | 否 |
| `test01` | `test01` | `Output/test_01`的测试序号 | 否 |
| `exp12` | `exp12` | 第12组科研实验，与模型版本无关 | 否 |
| `run001` | `run001` | 该实验下第1次独立训练运行（seed/数据/独立启动） | 严格续训否；独立新运行才加1 |
| `step002000` | `step002000` | 该训练谱系累计第2000个optimizer更新步 | 续训后随step增长 |

实例：

```text
Output/test_01/experiment_12/train_runs/run001/checkpoints/
  v1__test01__exp12__run001__step002000/
  v1__test01__exp12__run001__step004000/
  v1__test01__exp12__run001__step006000/
Output/test_01/experiment_13/train_runs/run001/checkpoints/
  v1__test01__exp13__run001__step002000/
  v1__test01__exp13__run001__step004000/
Output/test_01/experiment_13/train_runs/run002/checkpoints/
  v1__test01__exp13__run002__step002000/
```

## 判定规则

1. **绝不自动生成`V12`,`V13`,`V10`,`V11`**作为新模型版本。Experiment 12/13是`exp12`/`exp13`；算法为V1时一直用`v1`。旧V1残差版与新V1忠实适配采用`method_variant`、代码commit、谱系ID等元数据区分，不能把历史版本编号等同于实验编号。
2. **暂停/恢复/重新排队或工作台Job ID变更不产生新run编号。** 同一实验、相同可续训模型及优化器状态，`run001`继续在新step处保存。每次暂停恢复保留`resume_event_id`, `attempt_id`, `previous_job_id`, `resumed_from_checkpoint_id`及审计日志。
3. 独立随机种子训练、从官方权重重启或发生影响可比性的配置变化时，开新`run002`及新的训练谱系；禁止混入`run001`目录。单纯提高max optimizer steps仍为同一run。
4. 同步分布式/FSDP的model shards、optimizer shards、scheduler、RNG、sampler和meta归档在相同checkpoint目录。避免只命名单个权重文件，漏了严格续训状态。
5. `latest`、`best`保留为引用真实checkpoint ID的索引/软链接，不当成正式checkpoint名字；业务接口从索引解析唯一ID及hash后再加载。
6. 确认`checkpoint_index.csv`含所有五段字段、配置指纹、基座权重hash、代码commit、train job ID、retry attempt ID、准确step、保存校验状态。评估工件用独立`eval_run_id`并反向关联准确checkpoint ID。
7. 旧权重路径不挪动、不覆盖；新增`legacy_alias`映射到其真实来源，无法核实的一律标`unverified`。

## 工作台展示

工作台模型版本显示为`V1`，实验显示为`test_01 / experiment_12`，训练次数显示为`run001`，checkpoint显示规范完整ID。暂停续训按钮链接到原训练谱系和最近一个`COMPLETE`的checkpoint，避免出现多个伪造的新版本。
