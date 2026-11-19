# v1 进度交接

最新实现（2026-11-13，本轮双指标）：用户要求将B1K官方指标与原GR00T任务进度/成功率同时展示。已浏览B1K2025归档官方leaderboard/evaluation/submission及官方源码，核查本机实际安装__editable___omnigibson_3_7_1_finder.py映射/nvme3/shared/b1k2025/BEHAVIOR-1K/OmniGibson/omnigibson。TaskMetric.end_callback完整成功得1，否则按初始未满足/终局满足谓词占该方案全部谓词比例、多个方案取最大；learning/eval.py在终止时用info[done][success]统计成功。GR00T behavior_env.py已使用该TaskMetric；task_progress另为细分检查项历史OR累计，现有EpisodeWriter.success也对回合/动作chunk success取OR。因此不能伪造两种Q公式；保留原字段及公式，新增原生终局成功独立显示。正常成功即终止时两种成功率通常相同。

新增evaluation/scripts/b1k_metrics.py：terminal_success优先native_termination.success，其次新b1k_success；缺失保持None，禁止由Q==1/进度100%/旧success推断。summarize只统计valid回合、official-eval，分别报告Q和终局成功覆盖与缺失计数，mean_q_score及百分比、full_success_rate、valid_episodes、requested_episodes、metric_coverage_complete、scope=valid_completed_episodes。它是本地已结束有效回合均值，不宣称50×10官方总榜；官方缺交实例计0规则不擅自套到本项目历史运行协议。fast-debug正式指标None，无效回合排除，缺失不补0。

groot_eval_results.py EpisodeWriter新增b1k_success参数/JSON与CSV列，在reset前从terminal chunk info.success最后一项采集，native_termination.success存在时优先；保留原success OR及q_values max逻辑、终止策略、动作执行、模型不变。worker与public results.json分别加入b1k_metrics；groot_eval_output.py更新共享汇总；view_results.py load_runs从全量旧JSONL只读汇总并在API _episodes加每集b1k_metrics，不改写旧文件，已运行的旧仿真继续保留原进程，新采集字段下次启动生效，旧进程存下来的native_termination已够网页显示官方终局成功。

网页四主卡：B1K官方Q-score（百分比+0–1原值）、B1K完整任务成功率（终局flag覆盖）、GR00T累计任务进度、GR00T回合成功率。有效/计划/无效数量仍在卡片说明。列表小卡显示B1K Q/成功率及GR00T进度；逐集表分GR00T结果、B1K Q、B1K终局成功、GR00T进度；录像选择下五项显示两套口径及动作步数，移动端两列；缺失用未记录，配置里解释范围与覆盖。VLM与人工评分独立、上轮启停与人工功能保留。真实旧例V_0/baseline/groot_task21_1：GR00T进度34.7619%、B1K Q .0333333（3.3333%）、两个success均0%，Q和终局flag均10/10。不要将原34.8%再描述为成功率。

验收108项pytest通过：test_b1k_metrics（旧记录只读恢复、缺失覆盖、0/False、终局成功与OR不同、debug/invalid不混入、writer/public/viewer一致）及episode_results/q_score_viewer/viewer_ui/human_review/vlm_service/live_preview/web_launch/eval_layout；Node新增两套指标并列与原生缺失不推断检查。真实Chromium tests/b1k_metrics_browser.cjs通过汇总、逐集、录像和390宽手机检查；tests/vlm_service_browser.cjs回归也通过，启停写接口拦截，无真实服务操作。证据outputs/shared/diagnostics/b1k_metrics_20261113/{browser_result.json,official_and_groot_summary.png,episode_metrics.png,episode_metrics_mobile.png,deployment.json}。只更新evaluation/README.md和本handoff，README包含官方来源链接。

部署仅重载8765面板PID2931038→3018602；先保存/恢复5条评估job、14条A100策略job。A100 ready PID117072:5557、2714333:5556保持；A6000诊断仍PID2929322:8901 ready，未重启；8766 PID2584863未动，原训练/策略/仿真未收信号。后台自动同步将新Python模块送入web_live，下次运行用新源码，已运行快照不变。不要用上文历史PID替代实时检查。

最新实现（2026-11-13，本轮）：用户报告A6000 VLM离线，要求像策略服务一样手动启停、评估页可选诊断、录像回合选择后突出任务进度。按systematic-debugging定位：A6000原PID2808708一直健康监听8901，远端localhost和4090 curl --noproxy健康；只有网页urllib误走继承的HTTP/ALL_PROXY=http://127.0.0.1:12000，NO_PROXY的100.64.0.0/10并不被urllib识别。先补代理故障回归并观察失败，再将vlm_diagnosis._get_json/_post_json改为ProxyHandler({})直接内网连接，未改全局代理。健康检查和实际诊断请求均受修复。

新增evaluation/scripts/vlm_service.py，通过SSH只管理现有systemd --user VLMService.service，固定1024201092WYH@100.100.2.30、GPU0、8901、原Qwen3-VL-8B-Instruct。原安装仍/home/1024201092WYH/vlm_service，不迁模型或修改远端网络代码。状态区分ready/loading/stopped/failed，start幂等；stop核对unit MainPID的/proc/cmdline包含精确原vlm_service.py，拒绝关闭其他程序。stop使用systemctl而非pkill；原服务是transient unit，停止回收后由systemd-run --user --unit=VLMService --collect重建。没有自动启动/自动重启。view_results.py新增GET /api/vlm/service、/logs，POST /api/vlm/service/start、/stop；原/api/vlm/health保留。

网页“推理评估”左侧新增“视觉诊断服务”卡片：启动、关闭、刷新、日志。“评估结果→录像与相机”VLM面板增加Qwen/不启用诊断选择和启停按钮；选择本身不启动模型或自动诊断，单集/整轮按钮仅在选择且ready时可用，后端校验service_id，保留旧endpoint兼容。健康刷新同步更新按钮。单集诊断返回捕获run+录像身份，切换Episode不会被旧异步响应覆盖。人工评价仍可在无VLM/无录像时填写，原功能保留。

录像页下拉框简化为第几集/任务进度/结果，其下新增该Episode的任务进度、Q、结果、动作步数四项显著指标，保留0值和缺失区分，禁止套用整轮均值。mediaEpisodes合并录像身份时保留原Episode指标。正常有录像时隐藏计数和归档路径，精简相机说明；GPU/env/seed在回合行悬停及配置中可查。移动端指标两列。两份文档仅更新evaluation/README.md和本handoff。

验收：95项pytest通过（test_vlm_service、test_groot_viewer_ui/human_review/q_score_viewer/live_preview/web_launch/eval_layout）；新增代理GET/POST、生命周期状态、幂等start/精确stop/拒绝外来程序、真实HTTP路由、逐Episode指标含0、诊断开关与异步切集回归。真实Chromium tests/vlm_service_browser.cjs通过，所有启动/关闭测试写请求拦截，不改变真实服务；截图recording_metrics.png、episode_metrics_mobile.png、service_controls.png及browser_result.json在outputs/shared/diagnostics/vlm_service_20261113/。人工评价真实浏览器回归tests/human_review_browser.cjs也通过，无真实评价数据写入。

部署：仅精确TERM旧8765网页PID2882737→新PID2931038，已先保存并恢复当时4条评估记录、14条A100推理记录。A100 ready服务PID117072:5557、2714333:5556前后相同，原训练/策略/仿真未收信号，8766面板未改；deployment.json在上述目录。随后确认VLM批处理已complete、8901无ESTAB在途请求后，经新网页API真实关闭→启动诊断服务，8.4秒ready，原2808708→现A6000 PID2929322；重复start PID不变，报告lifecycle.json。VLM当前在线16.3GiB，此轮只重启了诊断VLM本身，不把部署记录中的旧VLM PID误当现PID。后续进程可能由用户手动改变，以实时状态为准。网页HTML实时读取，后端Python修改需保留记录后重载；不要操作原A100策略服务。

最新实现（2026-11-13）：用户新增A6000 VLM效果诊断后要求每条评估数据可人工结构化评分/备注、保存和删除。按逐Episode实现，未更改模型/训练/VLM推理机制。新增evaluation/scripts/human_review.py，枚举/评分/文本校验、真实Episode身份解析、原子JSON保存、读取、删除与列表；以run+episode_index隔离，旧未关联录像按完整相对路径hash隔离。保存到run/human_reviews/<sha256(key)>.json，记录source=human、Episode、实例/seed/GPU/env、created/updated时间。评分0–100、visual_progress0–100、behavior_class、failure_stage、target_touched/stuck三态、vlm_agreement agree/disagree/unclear、notes最多10000字；至少填一项，保留0/False/None。不会改官方results/episodes和VLM缓存。A6000镜像rsync无--delete，人工文件保留在4090对应run。

view_results.py新增GET/POST/DELETE /api/human-review及load_runs._human_reviews。result_viewer.html在VLM块下加独立“人工评分与备注”，无VLM或无录像的已结束Episode也可标注；成绩表增加人工评分/备注列。Map按run/episode保留草稿，周期refresh不覆盖dirty，保存捕获提交时身份、异步返回不污染当前另一回合。保存修改同一个人工记录，删除只移除所选人工记录，其他Episode/VLM/正式指标不变。只维护evaluation/README.md和本handoff两份文档。

测试84项通过：tests/test_groot_human_review.py（真实HTTP save/edit/reload/delete，跨run/Episode隔离、同名录像隔离、无录像、错误身份/非法分值，原results/journal/VLM字节不变），test_groot_viewer_ui/q_score_viewer/live_preview/web_launch/eval_layout。旧viewer Node测试在用户新加VLM轮询后一直挂起，systematic-debugging确认setInterval(refreshVlmHealth,30000)保持Node事件循环；只在测试harness stub setInterval并给subprocess15秒超时，未修改生产轮询。新增两项Node真实JS检查draft保存和异步切集；真实Chromium tests/human_review_browser.cjs通过，含desktop/mobile截图、VLM离线、保存/重载/删除；浏览器拦截写接口，没有给真实数据写测试评价。命令PLAYWRIGHT_BROWSERS_PATH=$PWD/outputs/browser-tools/browsers node tests/human_review_browser.cjs http://127.0.0.1:8765。报告/截图outputs/shared/diagnostics/human_review_20261113/。

部署：8765旧PID2855808忽略SIGINT，确认没有运行中的VLM批处理和8901在途连接后，先从真实API保存5条evaluation_jobs和14条inference_jobs及process_start，备份到上述verification目录，再只TERM该面板PID、以原参数启动新8765，未向训练/推理/仿真发信号。部署deployment.json记录新面板PID、恢复记录计数、同PID/端口推理就绪列表及新人工GET接口通过。旧8766面板属于另一进程，未改动；原VLM服务不重启。训练/推理固定快照不变，本机自动同步仍正常。用户需刷新8765网页后使用新功能。

最新实现（2026-11-12 21:57）：用户要求训练脚本支持更多冻结参数、全量和LoRA；已完成公共入口及网页接通，未提交正式训练或启停原训练/推理/评估。仍只维护README/handoff。本轮之前的脏工作区、其他窗口QUICK_ITERATION/RETRAIN等文档保留，勿清理。

新增experiment/finetuning_options.py（纯标准库解析，Web/CLI共用）及finetuning.py：模式action_full默认兼容旧冻结Eagle训练；full全模型；custom按组件覆盖；lora默认llm+action_dit低秩，非目标动作投影/vlln/memory默认全参数。7组件freeze/no-freeze：llm,vision,vision_projector,action_dit,action_projector,action_vlln,memory。LoRA可选前3骨干与action_dit的所有nn.Linear（排除lm_head）；类别索引式动作投影不能套普通Linear LoRA，可冻结/全参数。rank16/alpha32/dropout0默认，可配LLM最后N保留层（当前16）、non-reentrant gradient checkpointing、AdamW weight_decay=.01/max_grad_norm=1。model/modules/lora.py实现标准W+alpha/r BA，保存原weight/bias键；config.finetuning_lora记精确路径/超参，Gr00t构造时重建，完整save_pretrained保存/服务重载，无PEFT依赖。

关键实现：first_training.cache_sequence在训练且Eagle有可训练参数时缓存原始input_ids/attention_mask/三相机pixel_values，骨干放入task21_ddp.TrainableStep内部参与反传/DDP；slice_features按扁平相机列表3*B切片。完全冻结路径沿用原无梯度特征缓存。task21_formal.initialize→task21_diagnostic.load_model(finetuning=args)在载入FP32权重后设置范围，仅冻结骨干参数转BF16，可训练权重保持FP32，防止完整恢复/异步验证把主权重舍入；LoRA初始化前重设seed，避免新Memory模块初始化影响配对组LoRA随机数。Eagle冻结视觉encoder与mlp1分别eval。32决策/跨窗detach/记忆机制不变。训练/独立validator均转发所有选项；resume_manifest校验原冻结/LoRA/优化器超参；每run/<arm>/config/finetuning.json存实际参数计数/模块路径，ddp.json保留选项。旧task21_formal/first_training独立历史诊断入口不作为新参数公开入口，统一使用run_groot_train.sh。

train/dashboard.html/js/remote.py新增模式、组件开关、LoRA详细项、末N层、checkpointing/decay/clip；Web建议full/custom/lora微批1，full通用权重lr2e-6，不声称实测优化。页面由8765现有服务动态读取，无需重启；A100与6000Ada的既有自动同步已开启，分别更新各自web_live，不改正在运行的固定快照。A100真实HTTP preview full/lora/custom均返回正确冻结/LoRA值，未调用start；A100 web_live/run_groot_train.sh --dry-run full/双卡/独立验证通过，直接调用镜像内.sh须显式--python-path /nvme3/projects/memory_vla/envs/n1d6-p0/bin/python（镜像自身无环境）。

验收：155项通过（test_finetuning/test_task21_ddp/test_migration_training/test_single_baseline_launch/test_formal_resume及两个Web参数/推荐测试文件）；之后补充的Node表单实际序列化断言单测也通过。覆盖full/LoRA两CPU Gloo rank、尾窗长度不同、梯度累计同步；小骨干+真实动作头/记忆的HF strict reload和动作逐值一致；续训FP32参数逐值不舍入；freeze和LoRA初始输出一致、参数实际更新；改变LoRA/冻结/wd/clip的恢复被拒绝。仅原有Diffusers/Pyparsing弃用warning，node --check、公共CLI help正常。

真实离线GPU验证：本机GPU0原仅1.46GiB空闲，加载checkpoints/groot_n1d6_base及assets/datasets/b1k2025_groot_task0_v1_200（Episode10帧0），跑一次LoRA优化，无完整权重写入/原checkpoint改动。最终报告 outputs/shared/training_dashboard/verification/finetuning/real_b1k_lora_smoke.json：rank4/alpha8/dropout.05，目标llm,vision,vision_projector,action_dit，LLM仅最后1层，冻结动作投影/vlln，开启LLM与vision checkpointing；6,674,736可训练参数，loss.0072696437，梯度L2分别.0205991/.1694734/.08074035/.01287192，LLM lora_B有实际变化，峰值11.3012GiB，11.53秒含加载。input_ids[1,284]、动作padding[1,50,128]。这是接口/反传检查，不是模型能力或默认配置显存基准；A100四卡持续被旧训练占用，未共卡做全模型显存验证。探针正常退出，GPU0恢复1.46GiB，无新增常驻服务。

最新调查（2026-11-11 16:54）：用户仅要求诊断记忆评估慢，使用systematic-debugging，未改网络/训练代码、启停服务或请求在线get_action/reset。当前5560为A100 GPU0/PID3007839、Task40 Memory step3000，固定source deployments/V_1_task40_mem（旧residual/16，非最新direct/64）。诊断脚本/JSON在本机及A100 outputs/shared/diagnostics/memory_latency_20261111/。起初GPU1/2空闲，在GPU1离线加载同checkpoint+真实Episode400010帧0/8双样本三路256图像；满记忆Adapter分段10.10ms，Eagle35.89ms、动作头55.61ms，含CUDA同步；主特征[2,283,2048]、视觉[2,3,81,2048]。两轮各60决策，容量保持16、各44次merge，没有无界增长。独立Baseline117.24ms/Memory136–138ms，但同Memory旁路也139–140ms，不能量化因果增幅；没有在干净测试复现历史2.5倍RPC差距。

历史Task34记忆task34_eval02与baseline/task34_eval01各10回合，同实例seed、各47810步：RPC0.431666 vs0.172734秒（累计多774秒），vector step1.21982 vs0.826132秒（多1177秒），总run5596.8 vs3421.1秒。明显慢同时包含仿真；baseline identity unlinked、非同时同条件，保留归因限制。当前Task40匹配已完成6回合均38856步，Memory1821.70秒/Episode、Baseline1856.12秒，此部分未变慢；全部timeout，仍非10回合最终结果。运行中policy count0是主进程结束才写计时，不是零延迟。

重要新外部状态：16:48:49另一操作启动V_1_baseline_task0_34_40_train02四卡20000步/no-validation，从通用GR00T-N1.6-3B，rank PID2569006–2569009。GPU0 rank2569006与5560服务3007839共卡，16:54 GPU0约43.5GB/94%，GPU1/2约36GB/88–89%，GPU3另有他人3580727共100%。不要擅自kill或重启，用户本轮仅授权调查；这些训练不是诊断启动的。后半段memory_paced、model_paired、threads1报告全部与训练重叠，不能拿155.98→174.57或171.20→199.11ms作干净因果证据，也不能把112线程改1称主因。investigation.json明确标记干扰，所有诊断worker正常退出、无新端口，服务PID不变。下一步比较须隔离训练/推理GPU；本轮仅维护README与handoff，未动其他窗口RETRAIN等已有改动。

最新执行（2026-11-11）：用户要求训练页多选任务0/34/40联合训练、可选排队、增加通用GR00T起点、可关闭验证并设置划分比例。已实现train/dashboard.html/js：select multiple原生Ctrl/Command多选，提交tasks数组；enqueue开关；validation_enabled默认true、validation_percent默认10（1–50整数）；关闭验证隐藏禁用GPU和比例、不传验证GPU，所有200条/任务训练。多选联合一个模型，任务等比循环。页面显示总训练/验证及每任务数，缺数据需先准备；prepare按钮处理所选缺失任务。默认seed42/10%保留各数据已登记180/20，不修改原数据划分；新比例/seed确定性按完整Episode重分，本次config/data_split保留per_task、episode_tasks及counts。旧单任务API仍兼容。

后端train/dashboard_remote.py支持tasks/validation_enabled/validation_percent/enqueue，验证已登记数据、Episode隔离/编号唯一、全局窗口不超过每任务train数；多任务仅在该run/dataset合并元数据，data/videos软链接复用A100数据，不复制视频。run_identity保留V_版本_method_task0_34_40_trainNN。新增train/web_training_queue.py CPU持久监督器，每5秒检查所选训练及验证GPU，flock全局.gpu_queue.lock序列化派发；重叠GPU按created_at先进先出，非重叠独立；共卡仅忽略用户显式许可的PID+starttime，同步入口仍检查GPU。排队控制器调用唯一run_groot_train.sh，不是第二个训练实现；与取消同一锁关闭启动竞态，原取消核验支持web_training_queue.py，取消不删记录、前端隐藏。队列还检查直接启动而未建立GPU上下文的网页运行占用。队列异常写failed。jobs含waiting_for_runs/validation_enabled；详情无验证时仍从checkpoints/latest.json事务标记列出已完成的每步权重。

数据/训练范围修改：data/episode_split.validate_split允许显式每任务train/validation counts且总数200，旧配置仍严格180/20；launch_task21_ddp新增--no-validation，no validator且保留正常保存；task21_ddp沿用async分支保存/停止控制，但关闭validation_queue发布，模型/优化器/每rank历史仍每1000/结束保存。与--train-only不保存的短检查严格区分。无验证controller不读best.json，已验证模式保持原独立异步验证。网络结构、Eagle冻结、记忆机制未改。训练快照/运行/原服务未动；Task40仍在训练，原联合队列两组已被用户取消，未重排。

新增官方基础权重已真实下载并同步A100：/nvme3/huggingface/train_model/GR00T/GR00T-N1.6-3B，本机checkpoints/groot_n1d6_base，约6.57GB，来源nvidia/GR00T-N1.6-3B固定revision d0814e7ecb19202e7c8468b46098b0b7ef3a6d61。脚本train/prepare_base_checkpoint.py通过hf-mirror官方仓库副本下载（原HF直连不可达，环境代理失效），两分片按Hub LFS SHA256验证后rsync，A100再次完整hash，provenance记录在权重目录；官方B1K权重未改。分片SHA9710d31d331b79cfa229fd23605ba9ad47e207cb0f4c7722fe0bacdc666c8326/eb79bbd7893068897cda18c86ed91064e92e20e356dea97bc398bf9c8bf2fa35。核查1106项tensor keys/shapes全相同；基础processor的behavior_r1_pro配置存在且与B1K相同，statistics.json全字节相同，其他本体rl_info字段不同不影响B1K。官方模型说明没有保证预训练从未含B1K，网页准确标“通用预训练起点”，原权重改标“已经B1K微调”，不要承诺严格B1K未见数据。现网options已能选两种，默认新表单优先base，已有输入保留。

验收完成：78项Web/推理/数据/评估UI测试及训练侧22项含旧DDP等价、恢复及新比例/无验证launch通过（共100项）；新增Node真实JS参数测试覆盖多任务、排队、关验证禁用GPU/省略参数与600/0、开20%480/120；真实CPU队列进程使用fake nvidia-smi检查先等前序再启动无害子进程。A100真实HTTP preview及同步后的run_groot_train.sh --dry-run已验证联合600/0与480/120，生成真实3路数据chunk软链接，未执行GPU优化；记录A100 outputs/shared/training_dashboard/verification/multitask_web/{no_validation,validation20,summary.json,base_checkpoint.json}。末次接口检查基础3B Memory联合preview通过，下载后台3428077已正常退出，日志outputs/shared/training_dashboard/base_checkpoint/download.log；无新GPU任务。页面HTML/JS现有8765服务直接读取，无需重启；本轮未向评估/训练进程发送信号，末次API显示当前评估3371217运行中、Task40两组training（之前3267977由其他操作更换，勿复用旧PID）。只维护README/handoff；后续以本条取代旧“仅固定180/20/单任务/不支持新排队”描述。

最新修订（2026-11-11）：用户要求取消排队的任务从页面消失。真实API确认联合Baseline/Memory已被用户取消，两条status=cancelled，但原前端未过滤，导致仍显示历史卡片。只调整train/dashboard.js：refresh过滤cancelled，列表计数/版本任务分组同步消失；当前详情被移除时清除选择/详情并返回新建训练，取消成功提示在新建页显示。后台记录、日志、配置和训练编号保留，不删除，不改其他任务状态，不再次执行取消操作。JS由现有8765服务直接读取，刷新浏览器生效，无需重启后端或训练。新增Node真实脚本行为回归覆盖已取消配对隐藏、计数、运行及完成记录保留、退出旧详情；先复现失败再修复，52项训练网页及评估UI测试通过，node --check通过。README同步说明，仍只维护这两份md。

最新执行（2026-11-11 10:17）：用户说训练网页全是待转换待上传，授权缺什么补到A100，并问Task0_34_40排队原理。真实API核查4/50已ready(Task0/21/34/40)，其余46只有本机原始数据；不是缺当前联合数据。Task0/21虽ready但preparation not_prepared，因为早期转换不走网页；修复TrainingManager.options以remote ready覆盖本机历史状态，并在A100逐文件审计后补两个旧任务统一preparation/{data_split,ready}.json，不改原split、视频、parquet。新train/dataset_inventory.py stdlib逐Episode检查200parquet+600非空视频、所有meta、fps30、82/23维、seed42/18020隔离。当前四任务全部200/600，三任务组合600parquet/1800视频也真实确认。

已真实启动本机后台全量数据准备队列PID3286542（worker3286576在转换Task1），不是已经完成上传。入口train/prepare_all_datasets.py --root /nvme3/projects/memory_vla --host 172.28.62.17 --workers4，os.nice10/CPU转换/不占GPU；只单任务并行4线程依次补46项，按任务号1、2…跳过0/21/34/40。总状态/日志/进程/清单在outputs/shared/training_dashboard/datasets/all_tasks/{status.json,prepare_all.log,process.json,inventory.json}，每任务仍datasets/taskN/{status.json,prepare.log,config/...}。初始审计ready4、pending46、failed0，10:16 Task1已128/600视频。发布ready前prepare_web_dataset新增本机完整文件及远端上传后完整文件核查，worker参数可配；官方归一化和动作标签原样。失败单任务记failed继续其余，最后再审计全50；本机/网络须一直开着。不要重复启动队列，网页queued任务不再重复提交。用户授权补全部缺失，不下载已有原始数据，不把all_tasks数据队列误称训练队列。

网页数据下拉状态改为已就绪/数据排队/转换/上传，创建页每30秒刷新options保留已选参数，显示总就绪数、当前任务、600视频进度。后端options附dataset_queue；A100实际ready优先于本机转换历史。train/dashboard_remote.jobs新增从training/queues/*/status.json关联已预留job，附queue.dependencies/unexpected_pids/error/path，queued训练显示真实waiting_for_four_runs或waiting_for_gpus；日志配置页列前四组进度和待退出PID。只改当前开发源码及web_live，不动现有固定训练/联合快照，模型代码零改动。

联合训练队列PID740220仍alive，source与SHA未变；10:16依赖Task34 Baseline/Mem均3000且训练/最终验证complete；Task40 Baseline2127、M11712/3000，原PID3566627队列仍training/task40。每10秒严格等四组3000模型+最终验证，不依赖网页。两组联合数据已齐，完全不等待另外46项准备。重要：allowed_processes仍仅旧2676098:start1266488835，该原服务已自行退出；现在新增93858(Task34 Memory服务)、803592(Baseline5570)、100092等GPU进程不在许可名单。四组结束后会因这些进入waiting_for_gpus，已明确告诉用户并加入README，不能假称必然马上开训，也不要擅自杀服务或另一账号进程；本轮未调整许可或GPU映射。前一轮提到3957629属别账号的服务此时已不在GPU清单，不是本轮停止。

验收35项tests/test_training_dataset_inventory.py、test_groot_training_dashboard.py、test_groot_inference_launch.py通过，JS node --check通过；新增缺相机文件不能ready、旧数据从原run补登记且split不变、A100 ready统一状态、queuedjob依赖及GPU原因测试。真实HTTP确认Task0已ready/completed，Task1converting，Task2+queued，两个联合训练显示waiting_for_four_runs及逐组依赖。仅为加载新的TrainingManager方法SIGINT重载精确原网页PID3262022→3292212；原运行评估3267977:start170952560启动时间与HTTP恢复记录一致，其他训练/推理不动，证据outputs/shared/training_dashboard/verification/dataset_preparation_web_reload.json。仅维护README和handoff，不新建md。

最新执行（2026-11-11 01:08）：用户授权前四组Task34/40全部训练及最终验证完成后，从干净官方权重安排Task0/34/40联合Baseline+Memory，要求更多步、目标Loss约.0025，并追加提醒Task40排队。已真实提交A100持久队列PID740220，当前waiting_for_four_runs，原Task34/40队列PID3566627仍活跃。顺序：Task34两组各3000含最终验证→Task40两组各3000含最终验证→联合两组各18000；没有提前启动正式联合优化。01:08 Task34约640/540步，明确告知按速度联合明早很可能未开始，不能承诺明早完成或Loss达标。Task40已在原config.tasks=[34,40]，不是漏排。

**命名规则已被用户另一个窗口更新，别改回mem/bas旧规则！** 最新train/run_identity.py集中定义V_版本_baseline或memory_task任务号_trainNN；联合任务标签0_34_40。已保留两真实目录：V_1_baseline_task0_34_40_train01、V_1_memory_task0_34_40_train01，根/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training。当前旧单任务物理名及迁移兼容链接仍照常保留，原migrate_run_names.py/PID4016746不动；不要改现有固定源码或删除仍被队列/5560引用的链接。新增training_identity识别split.tasks组合标签，避免误归Task0；其余另窗口新命名工作保留。

新队列目录training/queues/V_1_joint_task0_34_40_train01，包含config/status/process/summary.json、queue.log、verification/；源码快照deployments/V_1_joint_task0_34_40_queue_train01，SHA d1c9b60286daab93050aa22e36d17cec976439669aec00dde2f7f86d3b6e3ea3。两训练各有deployments/同运行名快照，manifest一致，模型目录对比原四组固定快照无一文件变化；后续本机/web_live改动不影响排队运行。两个任务已保留queued arm/status.json，但root/status.json在实际launch前必须缺省（官方入口fresh-run检查）；不要手工补root状态。后台进程无需本机或SSH保持，不要重复提交。

两组统一官方/nvme3/huggingface/train_model/GR00T/GR00T-N1.6-BEHAVIOR1k起点、seed42、lr1e-5 AdamW wd.01 clip1、global4窗口=128决策/步、micro4、32决策梯度窗，Eagle冻结/原动作专家全参数训练/无LoRA，Memory仅既有结构。Baseline GPU0/1、validator0；Memory GPU2/3、validator3。每1000保存、每3000验证（另有step0官方参考），验证独立并行；--disable-early-stopping固定相同18000预算，不因独立验证平台期各自早停，也不靠偶发低Loss停止。可用原stop_training.json人工控制。预计18000总步每任务约6000次更新；不同Episode长度导致每任务完整pass长度不同，不伪称统一Epoch。

数据已在A100/nvme3/data_wyh/datasets/b1k2025_groot_tasks0_34_40_v1_600准备完成，以data/videos任务目录软链接复用转换数据，meta合并600 Episode；保留Task0原run划分，Task34/40准备划分，合计540训练/60独立验证。验证meta官方modality/stats/relative_stats/tasks一致、全局Episode ID无重复、每任务180/20无交集，最短episode970/1385/1584帧满足两窗，图像/动作/语言/归一化原样不变。split带tasks/episode_tasks/per_task，不带scalar task，避免catalog误将联合注册为单任务Task0。

实现范围只数据/采样/训练接口及调度：新增data/episode_split.py共享180/20×任务数检查；EpisodeWindows附task；TaskBalancedPlan按种子固定任务循环40→34→0，每任务独立pass，EpisodePlan仍保证同Episode固定slot→rank和顺序，按总position确定性重建resume各任务epoch/cursor，跨任务保留各自detach banks，仅各自Episode结束Reset。task21_formal.initialize接受540/60；错误历史donor仅同任务独立train Episode；验证记录逐Episode task和validation_by_task。DDP异步验证间隔可独立于1000保存，联合用3000；既有1000默认不变。plot新增per_task_curves.png；队列每30秒汇总逐任务100更新均值/目标观察/梯度/显存/吞吐，每5分钟绘图，训练时比较两组官方动作head SHA/预算/冻结/seed相同。队列依赖检查四组状态completed、training_complete、3000模型及validation_status含3000，适配原迁移映射；未知GPU进程等待、不杀进程，依赖失败报错停止接续。

验收：18项新采样/划分/donor/命名/四依赖门控及既有DDP/恢复/launcher测试，12项记忆/长序列/语言测试，32项训练/推理网页测试，共62项通过；最后plot/队列补充后17项核心测试再次通过。A100真实双卡GPU0/3短检查6步（不保存权重/不改变官方权重）覆盖40/34/0循环两遍，Loss finite、75项记忆梯度逐步全正、两rank全部参数SHA完全一致，峰值35.57GiB、每步约5.4–6.2秒；记录queue/verification/{summary.json,real_joint_memory6/...,real_joint_memory6.log,process.json}。最初预检被GPU0已有两个validator上下文拦截，通过systematic-debugging确认PID3655366/3667517是原Task34两个validator、各约.5GiB，再仅为短检查允许共卡；不是启动器，已核查cmdline。原其他GPU0任务2825251已自行退出，5560/2676098保持不变。新队列正式启动仅允许准备时捕获的既有PID+启动时间，四组结束前不会占卡；新GPU用户任务会等待。

用户明早查看：README顶部有完整cat/tail命令；先看新queue/status.json（waiting_for_four_runs是正常）、旧queues/V_1_task34_40/status.json确认Task40，再看两新run日志/plots和queue/summary.json。最低验证Loss仍仅候选，不称最佳机器人策略。只更新本README/handoff，无新增md；后续历史段落中旧命名、联合尚未实施和单任务限制以此条为准。

最新执行（2026-11-11 00:16）：用户要求改为V_1_mem_taskxxx顺序，并把本机/A100 training历史目录全部改名；询问A100数据、允许缺失任务从本机找。核查本机assets/datasets/b1k2025已有50任务全部齐备：10000 parquet约153.11GiB、30000三路mp4约346.92GiB；每任务200+三路各200已逐项确认，inventory在outputs/shared/training_dashboard/verification/dataset_inventory.json。A100可训练转换数据仍Task0/21/34/40，各在/nvme3/data_wyh/datasets/b1k2025_groot_taskN_v1_200；已检查A100 serf/serf_vla/serf_2025/b1k2026，不把2026不同版本数据混用。其余46任务可复用上一轮网页准备按钮，从4090已有2025数据转换并传A100，无需下载；没有额外复制约500GiB全量原始数据或启动其他任务。

已完成真实物理目录迁移，两端outputs/v1_groot_memoryvla/training（A100根为/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training）当前6个目录：V_1_mem_task0、V_1_bas_task0、V_1_mem_task21、V_1_bas_task21、V_1_mem_task34、V_1_bas_task34。原Task21配对root拆为独立两组，直接移动baseline与memory_v1权重/优化器，公共小日志/绘图/配置复制保留；各自ddp_launch/ddp_resume commands及selected_checkpoints只保留自身arm，JSON中的输出/发布checkpoint路径更新。已有部署源码目录不改名不改内容，原safetensors/.pt完全未重写；66个原文件inode/size/mtime_ns均一致，证据在A100 training/queues/name_migration/{weights_before.json,verification.json}并同步本机。Task0、Task21旧长名目录已消失或变为必要兼容链接，不要再往旧本机目录rsync制造第二份目录。

在线路径兼容：A100训练队列/当前进程仍在内存中使用旧输出路径，因此V_1_task34_bas→V_1_bas_task34、V_1_task34_mem→V_1_mem_task34、20261110_2005_task0_memory_v1_ddp_s42→V_1_mem_task0共3条旧名软链接临时保留。迁移仅在rename+建立链接期间对精确引用旧目录的进程SIGSTOP/SIGCONT数毫秒，没有SIGTERM、没有恢复/重启训练/评估/模型；2676098端口5560仍在。00:18只读复查官方2233562/5556已由其他操作退出，非本轮停止，未擅自恢复。不要手删这些链接。新增train/migrate_run_names.py已在A100后台PID4016746 --watch运行，记录training/queues/name_migration/{status.json,process.json,migration.log}；每10秒检查，Task40按旧固定队列启动后也会迁到V_1_mem_task40/V_1_bas_task40；队列完成且无进程引用时更新JSON路径并移除旧链接。Task0旧链接可能保留到用户自行停止5560服务，监视器届时自动清理。队列原PID3566627/源码快照/SHA保持不变，不要重复启动；兼容链接在队列完全结束前不会被清理。

网页/后续运行统一：train/dashboard_remote.py run_name改V_版本_mem或bas_task任务号，display_name同规则，catalog/jobs跳过兼容根软链接避免重复，detail已发布checkpoint返回resolve后的新路径。evaluation/scripts/groot_inference_remote.py checkpoint_info解析兼容路径到真实root并统一新显示名，但process_info的checkpoint身份字段仍保留真实进程原argv，保证停止服务的精确校验；checkpoint catalog跳过兼容根。训练RPC每次读最新源码；推理Manager在模块导入时缓存REMOTE代码，需重载网页后端才能更新推理选项。已保留评估记录SIGINT重载3023958，新8765后台PID3044699，真实确认评估PID/运行状态保持、5560服务仍ready且显示V_1_mem_task0；停止的历史服务标签也统一更新。记录outputs/shared/training_dashboard/verification/name_migration_web_reload.json。真实API只有6个训练记录、27个唯一checkpoint（含官方），Task21两组各10权重路径全部有效，Task40预检新名称V_1_mem_task40；72项相关测试通过（含别的窗口新增测试），最后推理标签补充后30项训练/推理测试通过，新增两项验证活队列兼容/清理与配对拆分不改权重/源码来源。输出index已拆Task21两项并全部更新新路径；README说明与可执行路径同步修改。旧交接中的名字均为历史，以本条为准。仅维护README和handoff。

00:16真实训练复核：V_1_bas_task34为190/3000，V_1_mem_task34为163/3000，原rank PID3683504/3676882不变，迁移后仍连续推进；GPU1/2、global128、micro4、seed42、lr1e-5、每1000保存、GPU0异步验证不变。Task40仍排队，不误报已启动。网页训练/推理列表通过新路径可用，原Task0本机评估和推理服务未重启；其他窗口可自行停止自己的服务，不应由本轮恢复它们。

最新执行（2026-11-10 23:59）：用户授权Task34/40各Baseline与M1共四组、各3000步；追加要求训练.sh及网页支持所有任务。已完成通用数据准备与网页全50任务，并真实启动Task34两组，Task40两组由A100持久队列等待Task34全部训练及验证完成后启动。不是多任务混合训练，不改网络，不从Task0/21微调权重续训。Task0 M1实际已3000训练/保存/最终验证全部完成（root.status completed）；其step3000独立验证Loss=.0034497272，正常/空/错误历史MAE=.0090142413/.0090859334/.0090122484，范围仅每独立Episode固定中间窗口，不能据此称已学会正确历史或最佳机器人策略。

四组新名称：V_1_task34_bas、V_1_task34_mem、V_1_task40_bas、V_1_task40_mem；输出均在A100 /nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/ 下。两组共用GPU1/2各双卡DDP，global_windows_per_step=4=128决策/更新、microbatch4、lr1e-5、seed42、3000步、每1000保存；GPU0两个独立validator并行。原计划考虑三卡global96，但用户新推理服务占用GPU3后统一改两卡global128，已明确告知，四组相同设置，不把它们与旧Task0 global96当相同曝光预算。保留GPU0原官方5556服务及其他进程；没有杀GPU3的5560/5561服务。本机评估PID2980985（web id d5e468299cda，Task0，evaluation/V_1/experment_1/groot_task0_1）也保留。其他窗口可能自行停服务/编辑UI，应读最新状态，不覆盖它们的工作。

持久队列：A100 PID3566627，目录training/queues/V_1_task34_40/{config.json,status.json,process.json,queue.log}；固定队列源码 /nvme3/projects/memory_vla/deployments/V_1_task34_40_queue，SHA a0444a249a1061c30604df10719fc7beddf055b94b6a55b5d3962c160e5944fd。每组再复制同一固定源码到deployments/各短名，调用唯一公共入口train/run_groot_train.sh。不要重复提交队列！Task34控制器PID3571048(Baseline)/3573605(M1)，23:59复查54/49步，Loss均finite、M1全部75项记忆梯度为正，单步约6.3/7.7秒（共卡初期快照，不作最终速度承诺）。GPU1/2总占用约74GB/80GB、利用率94–98%；每模型记录的峰值约33.13/35.86GiB。Task40尚排队，不能说已经训练。队列切下一任务前若有新增非授权GPU进程会标failed且不杀任何人，需检查queue.log；没有自动恢复/重启，已运行组不因队列失败被杀。

数据已真实完成两任务：本机assets/datasets/b1k2025_groot_task{34,40}_v1_200；远端/nvme3/data_wyh/datasets/b1k2025_groot_task{34,40}_v1_200，每任务200 parquet、600三路视频，preparation/data_split.json固定seed42、180/20无交集、stride8、gradient_window32，官方statistics保持。preparation/ready.json在传完数据和配置后才发布。原始数据来自本机assets/datasets/b1k2025已有SERF/共享路径，未修改原数据。准备日志/status/config统一outputs/shared/training_dashboard/datasets/task34和task40，无额外Markdown。

全任务实现：新增train/prepare_web_dataset.py（本机环境调用既有prepare_task21_formal.prepare(task=0..49)、rsync注册）；train/dashboard.py options返回50项、raw和三路视频数量、ready/can_prepare状态，prepare_dataset API后台启动转换；train/dashboard_remote.py catalog识别A100 preparation/ready.json；dashboard.html/js新增准备按钮与状态，未就绪不能启动；view_results.py POST /api/training/dataset注册。训练.sh原本就由dataset/split选择任务，没有21/0白名单，内部task21文件名历史保留，仍单任务固定200/18020，不支持混合训练。CLI不支持--task参数，--task属于数据准备程序，README已明确。

为加载新后端且保留正在运行的评估，view_results.py加入DetachedEvaluation只读进程观察器与PID启动时间校验，退出保存evaluation_jobs/inference_jobs JSON，下次恢复工作台记录；SIGINT重载仅旧网页后台PID2894169，未停止评估/模型服务，新后台PID3023958端口8765。训练源码自动同步仍enabled/synced/error=null；队列及已跑模型是固定快照，不随web_live改动。另一个窗口同时修改推理发现与网页布局，已保留；回归发现新增推理测试用pytest保留参数request，按systematic-debugging定位后仅改为stop_request，无产品逻辑改动。

验证：69项训练/推理/评估网页相关测试通过（PYTEST_DISABLE_PLUGIN_AUTOLOAD=1；系统ROS插件不兼容）；真实Firefox确认50任务、0/21/34/40可训练、34准备按钮不可重复、18020提示、6条旧新记录、34 M1 GPU1/2 global128参数预检通过，未借浏览器额外启动训练。记录及截图outputs/shared/training_dashboard/verification/all_tasks_{check.json,browser.png}。首次浏览器断言只因在异步jobs加载完成前读卡片，改为等真实卡片加载，产品无相关修改。仅维护本README/handoff两份文档，outputs/index.json已登记四组和队列；下文旧状态是历史快照，以本条最新内容为准。

最新执行（2026-11-10 23:15–23:19）：用户要求记忆组到3000步停止，后续名称按代码版本/任务/mem或bas，参考本机评估V_0/V_1分层。已通过现有网页stop API对Task0 M1运行20261110_2005_task0_memory_v1_ddp_s42设置stop_after_step=3000（23:15:35；当时2690步），远端metrics/stop_training.json已原子写入，并同步到本机对应run；没有杀进程。23:19复查2782/3000仍在训练，网页max_steps已3000；到目标保存后训练正常退出、异步验证继续。本轮只需排定停止，无需等待全验证完成；绝不能误报已经训完或恢复5000预算。原Baseline仍3000 completed，原服务均未操作。停止证据JSON在outputs/shared/training_dashboard/verification/memory_stop_3000.json，outputs/index.json已登记用户停止点与短名。后续恢复此run须先处理停止控制文件，不能自动续训。

命名已实现：新网页训练支持version参数（默认V_1，格式V_数字），train/dashboard_remote.py统一run_name得到V_1_task0_mem/V_1_task0_bas，同名目录或源码快照存在则追加_r2/_r3，避免覆盖；移除时间/长任务名/seed/UUID目录串。参数和SHA仍在config记录。旧run物理路径保留，jobs.name与checkpoint候选标签转换短名，页面列表/曲线标题已显示V_1_task0_mem、V_1_task0_bas、V_1_task21_mem、V_1_task21_bas；API run仍为原目录identity以保证stop/detail/checkpoint选取可用。不新建别名目录，不移动运行中的权重。网页代码版本字段只是运行标识；同步与执行仍当前V1工程，填写V_2不会实现/切换不存在的V2网络。CLI本来就用显式--output，手工启动按同样规范填写即可。本README的目录示例和使用步骤已替换成短命名规则，仅维护两份文档。

验证：57项相关测试通过（PYTEST_DISABLE_PLUGIN_AUTOLOAD=1），新覆盖默认/指定版本、错误版本拒绝、重复命名_r2、原快照保持不变、旧记录短名；JS语法和Python编译检查通过。实际Firefox浏览器验证4条旧记录短名、V_2_task0_mem预检，未提交新训练；记录outputs/shared/training_dashboard/verification/naming_check.json。自动同步已将代码送达A100 web_live，最新状态enabled=true/synced/error=null。无需重启网页后端，前端刷新可见新版本字段。注意上文/下文旧描述的长命名规则已由本条替代。

最新执行（2026-11-10 22:55）：本轮将训练控制接入现有评估网页，并开启4090→A100自动源码同步；没有提交新训练、恢复Baseline、重启推理或变更网络。现有网页 http://localhost:8765/ 顶部新增“启动训练”，原后端PID2894169（22:38启动）已经加载训练路由，本轮无需再重启它。页面可选Task0/21已登记200条单任务数据（180/20隔离）、Baseline/M1、初始化checkpoint、任意所选训练卡列表、异步验证卡、步数（1000整数倍）、microbatch、全局窗口数、lr、seed、显式共卡。窗口数必须整除卡数；32决策/窗，默认global3=96决策/更新；不静默杀现有进程。复用唯一公共入口train/run_groot_train.sh→launch_task21_ddp，Eagle冻结/无LoRA/网络原样；从所选权重新建优化器，不支持网页恢复AdamW/游标，旧CLI的--resume-step保留。

实现文件：train/dashboard.py（TrainingManager，本机SSH/rsync与4秒监听）；train/dashboard_remote.py（远端stdlib RPC：登记数据、预检、冻结源码快照、提交统一CLI、读取记录、保存边界停止）；train/dashboard.html和dashboard.js（表单/50步或完整Epoch Loss曲线/每10秒刷新/训练和验证独立进度/已发布checkpoint/带到推理页）；evaluation/scripts/view_results.py注册/api/training/{options,jobs,detail,preview,start,sync,stop}及静态资源，result_viewer.html增页签。既有评估/推理实现均保留。历史Task0源码从run_config/extension读取，Task21从remote_locations按arm读取，避免checkpoint跳转时套错源码。保存后停止只允许单arm运行与未来1000整数步；配对旧Task21不能单侧网页停止。

同步已真实启用：outputs/shared/training_dashboard/sync_state.json enabled=true，网页后端每4秒检测src/v1_groot_memoryvla/、train/、根pyproject.toml及uv.lock（若存在），单向rsync到A100 /nvme3/projects/memory_vla/deployments/web_live/，不传.venv、cache、数据、outputs、权重、evaluation。--delete仅限专用镜像中的两个源码子目录，不碰既有部署。每次点击启动先强制同步并检查SHA，然后copytree成deployments/task00_turning_on_radio_baseline_s42_时间_随机标识（M1对应arm）的固定快照，远端训练输出同名目录置于/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/。运行中代码不随本机编辑变化，改动用于下次训练。继续用A100既有envs/n1d6-p0/bin/python；依赖配置同步不自动安装。关浏览器不停止同步；关网页后端会停止，重启读取持久化开关。

验证：新增tests/test_groot_training_dashboard.py，包含mock远端Popen快照不变性/统一入口与全局batch/忙卡显式授权/训练与验证分离/加权Epoch均值/只选已发布checkpoint/单组停止范围/manifest排除环境/HTTP路由。与既有网页测试44项通过（需PYTEST_DISABLE_PLUGIN_AUTOLOAD=1，系统ROS launch_testing插件与pytest不兼容，不改项目依赖）。实际Firefox桌面1440与移动500宽度验证：4条旧训练记录、M1实时曲线、Baseline6个完整Epoch点、真实GPU1预检与global96、Baseline step3000→推理页checkpoint及正确source，均通过、无横向溢出。没有调用真实start，仅预检；A100统一训练CLI使用同步源码和既有Python --dry-run退出0，确认未创建output。自动同步后修改本README，观察SHA自动变化并核对A100全部137文件SHA一致，无.venv传输。记录与截图统一在outputs/shared/training_dashboard/verification/{browser_check.json,sync_check.json,canonical_dry_run.json,training_desktop.png,training_mobile.png,firefox.log}；源码snapshot通过mock测试，未为测试开实际训练。使用说明已写入本README，未新增Markdown。

最终复核：加入既有推理启动测试后共55项通过；最终源代码变更后单独复跑7项新训练工作台测试通过，JS语法与Python编译检查通过。自动同步仍enabled=true、state=synced，无错误，说明handoff修改也已自动送达A100；验证快照JSON是22:54时刻的证据，后续编辑会继续改变源SHA。

22:54只读快照：M1 Task0 2332/5000，GPU0/2/3仍训练；Baseline仍3000/3000 completed；原服务和训练进程未操作。GPU1已有用户新推理进程4108706，预检测试显式share_existing=true但未提交训练，不得误当空卡或杀它。Baseline step3000为用户指定闭环候选，最低验证Loss仍不能叫最佳机器人策略。

最新方案梳理（2026-11-10）：用户问单任务还是三任务一起训，本轮仅源码检查与方案说明，没有授权立即启动多任务或改实现，现有Baseline3000暂停/M1原预算均未操作。建议当前Task0闭环先确认推理/评估流程，再选三任务联合Baseline/M1；Task0为短流程对照，另2任务应有历史依赖证据，不能只挑最短3任务。分别训3任务是6组、适合归因；联合是2组同一共享模型、适合多任务策略，未证实更好/能修复退化。三任务各200条时每任务180/20固定、总540/60；32决策窗口按任务约1:1:1、顺时序/跨窗detach、task/dataset/Episode唯一键隔离，wrong history须同任务不同Episode+时间匹配；指标逐任务/逐Episode+任务宏平均，不只混合Loss。global96不变联合3000步每任务曝光约为单任务3000的1/3，要匹配约9000联合步；真实每遍窗口比取决于时长。

源码核查：prepare_task21_formal（位于data/非experiment/）一次单任务固定200、180/20；launch_task21_ddp与task21_formal.initialize写死180/20；EpisodeWindows super([dataset])封装单路径、by_episode裸eid（底层HistoryDataset支持多路径与dataset namespace）；EpisodePlan对全体窗口按长度balance、不保证任务采样平衡；validate donor任意train Episode，多任务需同任务筛选并按task聚合。后续需改数据/采样/验证调度，不改既有模型。README新增方案章节，未新增Markdown。参考查了NVIDIA n1d6 README和MemoryVLA论文（arxiv2508.19236）以确认多任务能力/记忆研究背景，不套当前main N1.7的新参数。

最新判断（2026-11-10 21:38）：用户问Baseline是否还需训至5000。只读确认3000验证已完整结束、controller/root.status completed steps_each3000，未恢复训练或变更M1（1075/5000，仍训练）。独立验证Loss step1000=.00341435477、2000=.00349244217、3000=.00346592030；固定动作面板MAE依次.00944524265/.00892071526/.00903884268。3000比2000验证Loss略低0.76%，MAE反而高1.32%，并无稳定持续改善，不能由此决定哪个checkpoint闭环更好。建议快速迭代保持Baseline停3000，先做同协议2000/3000闭环，没有现有证据必须继续5000；不是宣称3000已经收敛。3000 root.selected仍step1000仅验证Loss候选不称最佳策略。README补完整结果，原暂停条件保留，未动原5556服务和记忆组预算。

最新执行（2026-11-10 21:25–21:30）：用户明确要求Baseline到3000步先停，先评估并看Loss。收到时Baseline2952步，已在该run/metrics/stop_training.json写入stop_after_step=3000（原子发布），复用已部署async train正常保存后退出逻辑，无网络/训练源码改动、没有杀进程。21:28:55确认baseline/status.json training_complete=True/step3000，三rank均恰好1–3000无重复缺失、无3001；原rank PID2496583/2496584/2496585均退出，GPU2/3降到约38GiB（仅M1），GPU1仅validator约12GiB。模型step-003000、latest_optimizer.pt、三latest_rank-XXX.pt、latest.json（step3000/epoch6/position186）及validation_queue step3000均完整，真实保存32.77秒。原5556服务未动，M1继续训练，21:30到974/5000。Baseline controller/validator仍等待异步3000验证（3/20条），验证完成会自然退出、root.status完成，不能为释放GPU1误杀导致验证丢失。

权重：`/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/20261110_1745_task0_baseline_ddp_s42/baseline/checkpoints/step-003000`，评估请显式选择它，不自动选择best（当前仍step1000验证候选）。原始配置/日志total_steps=5000保留溯源，实际用户终点3000由stop文件与完成状态证明；outputs/index登记user_requested_stop_after_step3000和training_paused_validation_running。详细核查与Loss表生成在metrics/pause_step-003000.json，本机同步了轻量日志/配置/图片，无模型或优化器。若用户之后要求续训，必须先处理该run的stop_training.json，否则会在恢复下一步立即保存停止；resume仍需world3/global3同原设置，最新完整断点3000。不自动续训、不移除用户停止条件。

3000单步Loss .00169904303，但不与任意单步作收敛比较。完整epoch1–6各469步，加权均值依次.00353351043/.00328278113/.00305953379/.00297792375/.00288333244/.00281806727，最后一遍比上轮再降2.2635%，首末完整轮降20.2474%。第7轮2815–3000仅186步、均.00212067433，不能直接和完整轮比。3000独立验证尚未出，不能编造验证结果/闭环改善。用既有audit.plot_task21_loss刷新plots/training_loss.png到3000并同步本机；全首1000均.0033459747、末1000均.0027988473仅不同采样阶段在线统计。README写实际停止与完整权重路径，handoff本条交接，只维护两份Markdown。

最新只读估时（2026-11-10 20:45）：用户问2000是否能停、到3000多久。没有下达停止指令，不应自动修改5000上限。Baseline2368步，最近100步均4.0175秒/步，距3000纯训练42.32分钟；含保存建议约45–55分钟，完整独立验证另计。M1为436步，近100步均5.04999秒/步，距2000纯训练131.64分钟（约2小时12分）、距3000纯训练215.80分钟（约3小时36分）；对外按当前共卡速度给约3.5–4小时，不保证另一组结束/验证阶段变动后的速度。Baseline第5个完整epoch（1877–2345）Loss .00288333244，比上一遍.00297792375仍降约3.18%；因此2000只能作为快速迭代的预算停止点，不是已经收敛或Loss不再降低的结论。两组有效比较应同优化预算，M1尚未到2000。未停止/重启/改动训练。

最新只读检查（2026-11-10 20:39）：用户问当前3卡、任意卡数支持与Baseline是否下降很小。实际进程确认Baseline GPU1/2/3、world3、2286/5000；M1 GPU0/2/3、world3、371/5000，共物理4卡且2/3共卡。公共脚本train/run_groot_train.sh→launch_task21_ddp，torchrun nproc由--gpus列表长度决定；单/双/三/四卡均已有真实运行。限制global_windows_per_step必须能被world_size整除（launcher与train双检查），train硬上限world<=20、单机nnodes=1，当前global_windows=3故只能直接配1或3卡；2卡需可整除的窗口数如2/4、4卡如4，改变窗口数会改变全局batch/数据预算，不能宣称同预算；resume_manifest严格要求原world_size和global_windows，不支持本轮断点直接改卡数。不修改代码或正在跑的分配。

Baseline完整epoch0–3仍各469步，Loss .0035335104/.0032827811/.0030595338/.0029779238，逐轮降7.10%/6.80%/2.67%，最新完整一遍收益确实缩小，不能认定已收敛。epoch4当时只410步（1877–2286）、加权.0027792612，未完成，禁止与完整epoch直接比较。step2000独立验证刚出结果：Loss .00349244217，比1000的.00341435477高约2.29%；同时固定动作面板MAE .00944524265→.00892071526降约5.55%，两指标不能相互替代，更不能等同闭环策略能力。建议快速迭代可先用已保存1000/2000做同协议闭环，无须等Loss接近0；本次仅核查未停止/更改5000预算、服务或验证。M1当前step0结果Loss .00420798311、empty MAE .01204209330与官方Baseline相同、normal .01208505407；未到1000，暂不能与训练2000Baseline判断记忆收益。

最新查看（2026-11-10，用户问W&B及直观看Loss）：当前DDP/异步训练入口没有W&B调用、两个run也无wandb目录，仍是JSONL每步记录。只调用既有audit.plot_task21_loss在A100 CPU各刷新一次training_loss.png，并scp回本机同名run/plots；Baseline图覆盖2042步，M1图176步，已实际查看两张图片。上图50步均值、下图500步均值、灰线epoch边界，末尾不足50/500的部分分组也会画点，不能把Baseline2042尾点短窗骤降当全局改善；M1不足500步下图仅1点。两组要按相同优化步比较。README新增可点击图片及A100刷新命令；PNG是快照，不是实时面板，training_curves随验证完成更新、training_loss本轮手动刷新/训练结束生成。本次没有启用W&B、重启训练、改源码/参数或新增文档。核查时Baseline2033仍训练且独立验证step2000完成6/20条，M1训练169且独立step0验证到19/20条，均并行正常；图片生成后训练继续。

最新只读核查（2026-11-10，用户问快速迭代需要多少步）：Task0 Baseline已1967步、M1已107步。Baseline连续四个完整epoch各469步，按实际决策加权训练Loss依次.0035335104/.0032827811/.0030595338/.0029779238，首末降15.72%，最近完整epoch仅再降2.67%；step1000独立验证仍.0034143548，相对官方.0042052847降18.81%，2000验证尚未产生。M1当前不足一遍，不能推定其收敛点或记忆收益。建议快速迭代先以每组2000优化步（约4.3遍）为预算估计，1000用于首轮闭环检查，2000结果值得继续再延长至3000–5000；这是工程迭代预算建议，不是已确认收敛步数，也不要求Loss接近0。M1近20步约5.83秒/步，2000步按此速度纯训练约3.2小时，保存/验证另计且共卡速度会变化。本次仅答复预估，未修改两组现有5000步上限、早停规则、训练或服务；勿自动停止/缩短。

最新执行（2026-11-10 20:09）：用户询问4卡利用不足能否并行启动原记忆网络训练。已实际启动Task0 M1，不是只提供脚本；未重启现有Baseline/官方5556服务，未改网络或训练源码。新run=`20261110_2005_task0_memory_v1_ddp_s42`，远端`/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/`下同名目录；本地轻量镜像在根outputs/v1_groot_memoryvla/training/下。复用既有部署`/nvme3/projects/memory_vla/deployments/20261110_task0_baseline_resume_5000`；公共入口仍train/run_groot_train.sh。不要重复执行新run/config/start_remote.py，已有launch_process.json保护。

配置与当前Task0 Baseline公平对应：同官方原始权重`/nvme3/huggingface/train_model/GR00T/GR00T-N1.6-BEHAVIOR1k`、同Task0已转换200Episode数据、逐值相同180/20 split、seed42、原动作专家初始SHA、Episode虚拟槽及采样顺序、micro4/global3窗口=96决策/步、AdamW lr1e-5/wd.01/clip1、最多5000步。M1冻结Eagle、更新原动作专家及原MemoryVLA模块；perceptual256/mem16/retrieval2/时间编码/gate/update_fused=True/ToMe/32决策窗口/跨窗延续detach记忆均不变；无LoRA/Body Query/辅助损失，不继承Task21微调模型或optimizer。每1000步保存+独立异步验证正常/空/错误历史及逐Episode结果；初始step0也已交独立验证队列，验证不阻塞训练。本轮不自动启动闭环。

物理卡分配：M1三rank GPU0/2/3，M1 validator GPU0；原Baseline继续GPU1/2/3、validator GPU1；原5556服务PID2233562留GPU0。GPU2/3双训练共卡，实际总显存约72.3/80GiB；GPU0含M1训练/验证/服务约63.1GiB（会随阶段波动），GPU1当时仅Baseline约35GiB、验证时会上升。不是6张卡或额外独占3张卡；共卡会争用算力，不能承诺整体加速。原Baseline均PID2496583/2496584/2496585不变，controller2496370/torchrun2496571/validator2496572不变。新M1外层记录器2626732、controller2632044、torchrun2632411、validator2632415、rank0 PID2632607。

启动前一次4步真实共卡试跑通过，峰值allocated35.5411GiB、双路及门控/残差梯度有效，三rank所有可训练参数SHA逐值相同（83e8640f...），动作专家起点和Baseline SHA相同；证据新run/diagnostics/shared_gpu_preflight/及config/preflight_result.json。试跑不保存模型，正式组从官方起点重新初始化、计步从1，不继承4步试跑。20:09核查正式M1到24/5000，Baseline到1852/5000，全部已有Loss/梯度有限；M1正常验证同时完成4/20条Episode。关键梯度：perceptual q0=3.3235e-6、cognitive q0=8.6598e-7、perceptual gate=.00044352、cognitive gate=.0046442、perceptual residual=.0014959。M1最近10步均5.4369秒，Baseline3.8480秒（新增前约2.6秒），勿继续把旧Baseline独占速度当当前速度；M1仍需数小时，长跑随另一组结束/异步验证会变。官方文件size/mtime逐一保持；证据config/startup_verification.json。

新组看logs/memory_v1_torchrun.log（每步简洁日志）或memory_v1/metrics/metrics.jsonl（完整记忆梯度）；logs/memory_v1_validation.log是验证checkpoint编号，不是训练步；logs/controller.log为调度，logs/preflight.log为一次共卡短测调度。模型在memory_v1/checkpoints/step-001000至005000，候选best.json仍仅验证Loss记录。config保存run_config/data_split/ddp_launch/source_sha256/official_files_before/remote_locations/startup_verification；两组输出分开、未删原训练。README新增GPU分配和完整查看命令，outputs/index.json登记。只维护README与本handoff，未写新Markdown；本机镜像是当时快照，不自动实时同步，不拉模型。

最新只读解释（2026-11-10 19:48–19:49）：用户问Loss到多少算够、为什么上下波动。使用systematic-debugging，只读当前Task0日志与实际模型/采样/验证源码，无训练或网络改动。1383步快照：step连续不重不漏，Loss min/max0.0005450137/0.0148184173，NaN/Inf=0、zero-grad=0，max preclip-grad0.3582779、>1=0；初训与续训正常继续，总目标5000不变。完整epoch0/1均469优化步、1406真实窗口/44992决策，实际决策加权Loss0.0035335104257→0.0032827811278降7.0958%；简单每步均值0.0035333205868→0.0032818867465约7.116%，尾步仅2窗口因此最终报告以前者为准。epoch2当时仅445步，不能与完整epoch直接比较。

波动证据：每遍顺时间轮询Episode窗口、每步仅3条Episode的32决策片段，并按step/slot重采样Gaussian动作噪声+Beta时间，梯度更新没有每步全数据Loss单调保证。epoch边界470/939，前两遍100步分块呈相似低→高：1–100均值.002568、301–400均值.004278；501–600均值.002569、801–900均值.004274。不能以任意首尾100步判断退化；完整遍历有下降，但沿途更新/噪声不同仍不是固定样本/固定噪声重评。独立固定验证20Episode：step0 .0042052846885→step1000 .0034143547675降18.808%，仅两个点，不宣称收敛或闭环改善。Loss为模型gr00t_n1d6.py203–208与249–251的masked flow-matching velocity MSE、有效动作维度归一化，不是MAE/物理距离/成功率。答复应明确没有0.001等通用达标阈值，关注整遍/平滑训练趋势、独立固定验证是否改善，以及同协议闭环。未启动额外重评/调参/停止训练；README补证据，仅维护两份Markdown。

最新执行（2026-11-10 19:32）：用户要求Task0从1000延长到约5000步，另记录4卡相对1/2卡的速度。已完成真实1/2/4 A100公平短测，并实际从1000模型+AdamW+三rank游标续训到总计5000（新增4000），三rank均已核查1019/5000，Loss/梯度有限且非零，step无重复/缺失，total_steps新行=5000。仍是同一run`20261110_1745_task0_baseline_ddp_s42`，GPU1/2/3、global96决策、micro4、lr1e-5、冻结Eagle、无记忆/LoRA、180/20 split与seed42保持。1000模型目录保留，后续2000/3000/4000/5000各保存、异步独立验证共享GPU1。当前约2.66秒/步，剩余约3小时训练，另有保存/验证开销；未开启闭环。

续训部署`/nvme3/projects/memory_vla/deployments/20261110_task0_baseline_resume_5000`，初始Task0部署和Task21历史部署不改；model/所有Python SHA与Task0起点源码完全相同。仅改launch允许单组resume、resume_manifest允许总预算延长（不能缩至原上限以下或不超过断点）、异步启动在spawn验证器前把旧training_complete重置为False。后一项已用失败测试复现旧完成状态令验证器提前退出的竞态，按systematic-debugging定位修复，完整33 tests passed（两个新回归）。公共入口仍是train/run_groot_train.sh，A100默认源码更新为续训部署。本轮没有改网络/动作标签/优化器设置，也未覆盖原始权重。

原1000训练和验证已完整结束，官方step0独立验证Loss0.0042052846885414965、step1000为0.0034143547675090586，仅为拟合指标不能推断闭环成功率。旧控制器/验证器均自然退出后才启动测试与续训，无重叠验证器。恢复checkpoint epoch2/position62，restore_rank校验AdamW每条state.step=1000；各rank实测从1001开始。新控制器PID2496370，torchrun2496571，validator2496572；外层顺序测速/续训记录器2486250。validator_ready processed_checkpoints=[0,1000]，等待2000，不重算旧验证；GPU0仅官方服务PID2233562/5556，其他用户训练已自行退出，未被我们停止。官方文件逐一size/mtime与初训前保持相同。

速度实测：同Task0官方起点、固定4个虚拟Episode槽/每步128决策、micro4、seed42、相同前8步、冻结Eagle+原动作专家全参、不开验证/保存；排除前2步预热，统计6步，含读取视频/Processor/Eagle/反传/同步/AdamW。单卡GPU1平均8.0591525秒/步、15.8826决策/s；双卡1/2为4.5641411秒、28.0447决策/s；四卡0/1/2/3为2.6127919秒、48.9897决策/s。四卡/单卡=3.0845×、四卡/双卡=1.7468×，双卡/单卡=1.7658×，四卡效率77.1%；每卡峰值allocated约33.1GiB（不含官方服务）。四卡短测允许共卡的精确PID2233562、服务始终保留；短测仅6计时步并非长跑保证，各组rank参数SHA完全一致，无额外模型/optimizer文件。正式续训保留3卡96决策以准确恢复批量与游标，不混作4卡128决策训练速度。

测速原始JSONL/控制台/配置/哈希与summary.json在本run/diagnostics/scaling_1_2_4cards/{world-1,world-2,world-4,logs,summary.json}，README有速度表。准确续训argv在config/extension_5000.json、ddp_resume_step-001000.json，源码SHA在source_sha256_resume_1000.json，训练恢复标记在baseline/config/resume-step-001000.json。原config/run_config.json和baseline/config/ddp.json仍记初训1000步，保留作溯源，以extension和实时status为当前预算；勿按原start_remote.py重复启动。训练/验证日志继续追加同名文件，外层日志logs/controller_extend_5000.log。绘图在后续验证点及结束时更新；本地镜像仅快照，不含模型/optimizer，没有自动同步。只维护README及本handoff，不另建Markdown。

最新执行（2026-11-10）：用户要求从未训练的官方权重针对Task0再训一次、先无记忆baseline、可多卡并尽快。已实际启动A100运行`20261110_1745_task0_baseline_ddp_s42`，不是仅交付脚本。远端根目录`/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/20261110_1745_task0_baseline_ddp_s42`；数据`/nvme3/data_wyh/datasets/b1k2025_groot_task0_v1_200`；源码新快照`/nvme3/projects/memory_vla/deployments/20261110_task0_baseline_ddp`；起点`/nvme3/huggingface/train_model/GR00T/GR00T-N1.6-BEHAVIOR1k`。Task0共200条原始示教，全部转换、核对动作sha/帧数/30Hz/82维state/23维action/600视频，官方统计不重新拟合；seed42固定180训练/20独立验证，划分与转换明细在config。未继承Task21微调模型或optimizer。

本轮参数：无memory、无LoRA，冻结Eagle/eval，原动作专家全部1418624000参数更新，AdamW固定lr1e-5/wd.01/clip1，GPU1/2/3三rank NCCL DDP，global_windows=3即96决策/优化步，micro4、每窗32决策、stride8，最多1000优化步，仅保存step1000。Task0训练1406个完整窗口，每遍469优化步，约2.13遍。GPU0有他人的训练PID2150722及用户官方5556服务PID2233562，因此选择3张空闲卡，未操作GPU0或旧Task21文件。GPU数说明已在启动前告知。

18:40实际核查三rank均到64/1000步，所有已记录Loss/梯度均有限且梯度非零、memory_gradients={}；最近20步平均3.278s、29.42决策/s（约55分钟训练，另需末次验证，不能当作完成承诺）。训练18步时独立validator正在step0第4条Episode，训练54步时验证已完成10条，证明训练与验证并行。GPU1约47GiB（训练+验证）、GPU2/3各35GiB，GPU0两旧PID和5556监听不变，官方所有文件size/mtime逐一与启动前相同。外层记录器PID2304870，控制器PID2304872，torchrun PID2304981，validator PID2304982；子rank由torchrun管理。后续先读status与metrics，不要重复启动已有config/start_remote.py。已生成首版plots/training_loss.png（54步快照）和training_curves.png，已镜像本地并查看图像，最终会自动更新；本机镜像目前是快照，没有后台自动同步。

实现改动只在数据/调度/绘图，没有网络变化：prepare_task21_formal增加--task（默认21），单组launcher允许async-validation-gpu，并把独立验证卡加入占用检查；plot及plot_task21_loss兼容只有baseline。新增3项单组异步回归，完整31 tests passed。公共入口仍只有train/run_groot_train.sh，A100默认源码更新为新快照；原部署resume_r4及LayerNorm修复快照保持原样。step0官方权重先入独立验证队列，step1000保存后进入同队列，不阻塞训练。两次验证使用同20独立Episode全窗Loss和固定动作面板；baseline没有历史，因此normal/empty/wrong是同一路径。best.json可能仍指向官方step0，仅是验证Loss候选；用户评估训练后权重时应明确选baseline/checkpoints/step-001000，不能自动把best称为最佳机器人策略。

输出：每步baseline/metrics/metrics.jsonl（step,total_steps=1000,Loss,gradient_norm,显存和吞吐）；训练控制台logs/baseline_torchrun.log；验证logs/baseline_validation.log；调度logs/controller.log。结束后控制器生成plots/training_curves.png，外层记录器再生成plots/training_loss.png。config保存启动命令、run_config、source_sha256、official_files_before、ddp_launch和data_split；本机镜像不回传模型/optimizer。没有开启W&B、没有启动新的推理服务或闭环评估。下一步确认本轮完成和step1000能加载，然后用户可用同Task0实例/seed将官方权重与step1000配对评估；不能仅凭训练/验证Loss判断微调退化。当前共享评估入口已由其他工作接入0–49全部任务，本窗口只读确认支持Task0，未再改评估代码。只维护README和本handoff。

最新路径核查（2026-11-10）：用户询问未训练权重位置，已只读确认A100官方目录 `/nvme3/huggingface/train_model/GR00T/GR00T-N1.6-BEHAVIOR1k` 存在config/processor/statistics及两份safetensors。5556未监听，5560/5561仍监听原训练后两个服务。交付现有evaluation/scripts/run_groot_server.sh显式加载官方权重的命令，用户自行启动；未启停模型或评估，Task0入口尚未接入的状态不变。

最新短任务检索（2026-11-10）：用户要找步数最短任务以快速区分推理与评估问题。已读取本地tasks/episodes元数据，按任务指令精确匹配全部10000Episode、50任务各200条，无未匹配；平均最短Task0 turning_on_radio 2149.64步、中位2109.5、min970/max4353，其次Task34 hanging_pictures 2390.17、Task40 make_microwave_popcorn 3237.775；Task21 19186.325、Task22 7692.45、Task26 26060.055。Task0比21短8.925倍。核对真实runtime behavior_env.py 334–351行从gm.DATA_PATH/2025-challenge-task-instances/metadata/episodes.jsonl按episode_index//10000算均值，550–552行为int(mean*2)；该运行时元数据与demos统计一致。Task0 timeout4299（实际记录可4300），34=4780，40=6475，22=15384，21=38372。available_tasks.yaml已有turning_on_radio，底层环境支持；公共run_groot_eval.sh task choices、task_names和run_groot_priority_eval.sh TASKS仍只21/22/26，此次未改脚本/白名单、未启停仿真或模型。可直接用既有入口的最短是22，要用0需接入白名单/名称。建议诊断时原生官方服务与v1服务都加载相同未微调官方checkpoint，用单环境同实例/seed/action8/原生timeout；Task21微调模型在Task0差不能直接证明推理有bug。web核对官方模型卡支持BEHAVIOR/sim wrapper、2026官方eval默认已变1.5×而本地2025明确2×，勿混用。README补证据，仍只维护两个Markdown。

最新只读闭环退化诊断（2026-11-10）：用户问训练后两组任务进度34.8→17.9原因，已读并使用systematic-debugging，未改代码或启停任务。读取真实结果：V_0/baseline/groot_task21_1为single1、10完成、mean_progress34.76190476、success0、原5556；V_1/baseline与experment_1/groot_task21_1均single2、各4/10、mean17.85714286、success0、全部38373步timeout无invalid，仍running。新4实例0/5/1/6，种子0/1/1/2；旧对应0/5/1/6种子0/5/1/6，故不能直接做完整配对。instance0 seed0旧42.85714→两组23.80952；instance1 seed1旧38.09524→baseline9.52381/M1 23.80952，确有值得调查的退化信号，不能用回合未完搪塞。protocol hash不同源于evaluation.md字节摘要，本身不证明语义改变。

远端精确服务核查：PID2038009/M1 step010000/5561/GPU2，PID2038509/baseline step010000/5560/GPU3，均12:00启动、seed42、sim wrapper、resume_r4 cwd/PYTHONPATH，stdout是用户终端，未操作。不是selected_checkpoints的M1 3000/对照2000。两组checkpoint统计文件SHA与官方完全相同；embodiment_id文件内容有新增其他本体但BEHAVIOR均24；processor kwargs只有language_instruction_mapping和letter_box_transform新增，task21正确映射Collecting childrens toys.，letterbox=False与原构造默认相同；动作头tune_projector/diffusion/vlln均True、horizon50/max_dim128/推理4步，与原配置一致，处理器有效horizon32/23维。去docstring后AST对照动作头forward/get_action与参考相同，差异仅Beta显式FP32及已修的vlln函数返回副本等；旧double-vlln只解释离线normal MAE，不能解释闭环退化。训练split180/20无重叠；原valLoss早期2000/3000候选优于10000约1.2%/.76%，不可据此指定最佳策略。两组共有动作专家全参微调/小批量单任务/闭环分布差异值得优先查，但过拟合/灾难性遗忘未获因果确认。还缺官方权重经同v1服务+single2同实例seed的闭环桥接比较与早期checkpoint闭环；此轮未执行任何新模型/评估或改lr/LoRA/网络。README补同样证据及未确认边界，未新增Markdown。

最新文档整理（2026-11-10）：用户要求像评估命令速查一样列出训练常用指令和逐行参数注释。已将README统一训练入口段落改为A100公共路径设置、paired 2+2异步验证、baseline双卡、M1双卡三个可复制示例（均dry-run），补全当前真实支持的参数注释、单卡改法、输出日志位置；明确single-arm现有模式串行验证、train-only不验证也不保存checkpoint、每步32×global_windows决策、保存1000/异步验证1000/单组验证500为启动器固定设置，当前无interval/W&B用户参数；说明checkpoint模型起点与resume-step完整最新断点的区别。只修改README/handoff两份现有Markdown，不改脚本、训练器或部署，不启动任何任务。

最新交付（2026-11-10）：按用户要求统一训练用户入口为 `/nvme3/projects/memory_vla/train/run_groot_train.sh`，本机及A100均已部署、可执行且SHA256一致。薄封装现有launch_task21_ddp，单组M1/对照、单/双卡和paired均透传原CLI参数，不改网络或训练行为；支持--source-dir/--python-path覆盖，无参数显示帮助。默认本机workspace+v1 .venv；A100无根src时选deployments/20261110_mae_layernorm_fix（已含LayerNorm指标修复）+envs/n1d6-p0。历史resume_r4快照未改。README新增唯一用户入口、两组2+2卡和异步验证完整dry-run示例及项目树；说明单组现有模式默认串行验证，paired需显式传async两个GPU参数。完成本机/A100 bash语法与帮助、A100从/tmp运行无参帮助、paired 2+2/10000步/global64/保存验证1000/异步标记dry-run、两个单组单卡dry-run、源码/Python覆盖核查；dry-run均未创建NEW_RUN_NAME输出目录。此次未启动训练、模型或评估，未改既有权重/日志，也未加新Markdown；内部Python训练/验证模块保留作为实现，不需要用户分别启动。

最新交付（2026-11-10）：用户问训练Loss曲线/W&B/训练方法参数入口。新增audit/plot_task21_loss.py（只读JSONL，无模型推理），生成本轮plots/training_loss.png，分别50/500优化步分块均值、epoch切换6631虚线，已检查真实两组各10000条并查看图像；不替换旧training_curves.png，不包含旧MAE。真实A100 r4四个执行入口均无wandb调用，本轮无wandb目录/*.wandb，本轮未启W&B且未补传外部服务。README补单独曲线及完整训练方法参数表：DDP调度launch_task21_ddp.py/实际train task21_ddp.py/异步验证task21_async.py，非通用Trainer/train.py；lr1e-5固定、wd.01、clip1、micro4每rank32决策8块累积、global64决策、两组双卡、Eagle冻结动作专家FP32全训BF16 autocast、记忆配置如原。实际r4恢复完整argv为config/ddp_resume_step-001000.json，旧launch配置M1 eval500已被恢复1000取代。此次只新增绘图和两份文档，未启训练/模型/评估、未加入W&B或改训练配置。PNG同步A100同run/plots；训练源码快照未修改。

最新Loss诊断（2026-11-10）：用户问首尾降幅、为何回升、是否训练异常；使用systematic-debugging，只读统计/源码，未改训练。两组train均10000条无重复step；首尾单步降幅对照60.648%/M1 61.790%但不同样本噪声不可当收敛率。首尾各1000步均值对照0.0032939718837→0.0028314884921降14.0403%，M1 0.0033082087535→0.0028350929730降14.3013%；首尾500步降2.0981%/2.4725%。更关键：重建两rank日志的epoch/Episode窗口计数，严格assert每遍前3240位置恰好同6480个窗口、180Episode、无重复；在线均值第一遍→第二遍对照0.0031020411369→0.0030191927437降2.67077%，M1 0.0031160597687→0.0030205104486降3.06635%。不同噪声且模型沿途更新，仍非固定checkpoint重评。

EpisodeWindows.order按窗口序号轮询Episode，EpisodePlan两槽；6630步/遍，第6631步epoch1，10000约1.51遍。后段回升与重新访问前部片段吻合，baseline/M1逐步loss相关0.99805994，同step rank0 Episode全10000一致；7001–8000两组均约0.00331。GR00T原flow-matching随机噪声/时间点MSE，非动作MAE，seed按step变化。所有loss/梯度有限，两记忆库及残差每步非零梯度；global clip1，preclip>1仅对照13步/M1 7步（对照最大13.3757/M1 2.1266，未持续爆炸）；lr固定1e-5无scheduler，Eagle冻结。结论：没有观察到发散/断梯度，但同样本在线改善仅约3%，不能宣称充分收敛；要精确区分训练收益需固定训练窗口/噪声，对官方起点及保存checkpoint只读重评。本窗口仅回答与更新两份md，未启动该重评、未提早重训或调参。

用户最新判断口径（2026-11-10）：要求提供训练曲线并明确不能把最低验证Loss称为最佳机器人策略，最低验证Loss差约0.53%不足以判断实际任务能力；重点查看训练Loss。已核对本轮plots/training_curves.png本地存在256K，左上训练曲线每50步均值、两组各10000条train日志完整。末尾9501–10000步均值对照0.0028560319737944155、M1 0.002858513799654702；最后1000步分别0.0028314884920750956/0.0028350929730133886。曲线接近，不将train单步最低或val最低视为最佳策略；训练Loss衡量拟合/收敛、闭环衡量任务能力，跨step数据片段不同需同预算/同数据进度比较。README已修正当前表述，selected_checkpoints.json/best.json原样保留为历史验证指标候选，不自动重选/启评估。原图右上normal MAE含旧double-vlln bug，不能据此比较，修复数据见下方重算记录。此次只查询、更新两份文档，未改训练/服务/权重/指标记录。

最新修复（2026-11-10）：按用户要求修复LayerNorm并用已有checkpoint重算MAE，没有重训。根因是动作头process_backbone_output原地替换特征，formal.fused_window在Loss后收集normal，随后get_action再次vlln；empty/wrong只一次。现动作头返回新BatchFeature，调用者保持pre-vlln。新增test_action_normalization.py，修复前两个用例失败，修复后v1全套28 passed（Gloo须沙箱外本机回环网络）。新增audit/recompute_task21_mae.py，仅eval/no_grad/requires_grad_false，无optimizer；源码独立部署20261110_mae_layernorm_fix，旧resume_r4及服务源码不改。

重算选定M1 step3000、对照step2000各20独立Episode，沿用原split、每Episode固定中段32决策、完整preceding normal memory、时间匹配wrong及seed。M1 normal/empty/wrong完整32步MAE=0.008913138939533383/0.008941097522620112/0.008914899884257466，前8步=0.007045946642756462/0.007112921355292201/0.007049046061001718；对照32步0.0091835783328861、前8步0.007268260442651808。normal旧0.028383875358849765是假性偏高：20条empty/wrong与旧数组逐值一致，全部旧normal通过显式double-vlln逐值复现。normal相对empty/wrong仅改善0.313%/0.0198%，Episode胜出12/20与10/20，不宣称正确历史有效。验证Loss、选模与训练文件保持原样；其他M1步骤MAE仍含旧bug，尚未重算。

真实动作推理审计：所有20条Episode三个条件共享同一get_action、每调用1次vlln和4次DiT、相同microbatch2/BF16/RNG/state/embodiment/mask及形状dtype，输入特征不被修改；每分片相同特征控制输出逐值相等。各worker前后action_head SHA相同，checkpoint文件size/mtime未变，所有参数无grad。GPU1/2/3分片7/7/6条，GPU3另算对照，四worker全部完成并退出。新结果在本轮run的mae_layernorm_fix_20261110/{summary.json,launch.json,memory_v1,baseline,logs,merge_task21_mae.py}，已完整镜像本地，旧指标/曲线保留，不覆盖。本轮未请求、停止或重启5555/5556，也未使用GPU0；最终两服务已停，与下方另一操作按用户要求停止的交接记录一致。

最新只读核查（2026-11-10 10:03–10:04）：用户询问四卡占用。此前GPU1/2/3的PID1924798/1924799/1924800实际为audit.recompute_task21_mae离线动作误差重算，部署cwd为deployments/20261110_mae_layernorm_fix，产物为本轮mae_layernorm_fix_20261110/；未操作这些进程。初次快照仅GPU1/PID1924798仍跑M1 shard0约10.5GiB、64%，GPU2/3已退出。随后再次核查，四卡compute进程均空，GPU0/1/2/3仅35/32/32/32MiB、利用率均0%。日志summary确认对照20Episode完成，M1三个分片7+7+6Episode全部completed，training=false。旧5555/5556已停，新5561/5562尚未启动；此次未启停任何进程或修改评估代码。

最新运行状态（2026-11-10 10:00之后）：用户要求暂停“之前那两个评估”，准备改用训练后的权重。4090已无仿真评估/队列进程、两卡无compute进程；结合上文两个旧服务的指代，已明确告知用户将停5555/5556服务，并核对精确PID、官方model-path、port后执行。5556/PID905468响应SIGINT退出；5555/PID4055792未响应SIGINT，核对后SIGTERM退出。最终5555/5556/5561/5562均未监听，旧两个PID均不再占GPU，GPU0释放约13.84GiB。未删除任何权重/日志，未启动新的模型或闭环评估。A100 GPU1/2/3另有PID1924798/1924799/1924800各约10.5GiB占用，不属这两个旧服务，未操作。README更新当前旧服务已停；下方“原服务保持运行”为此前历史状态。

最新修订（2026-11-10，取代下方上一版启动器用法）：按用户要求，本机和A100入口均移为evaluation/scripts/run_groot_server.sh，原scripts/run_groot_server.sh已移除。取消--model所有预设及自动选模，三个必填参数--checkpoint完整目录、--gpu-index、--port；统一v1 serve按checkpoint memory_enabled加载对照/M1，B1K sim wrapper默认启用。保留可选source-dir/python-path/log-dir/seed/foreground/dry-run。默认resume_r4源码和n1d6-p0环境；训练权重日志归所属run/logs，文件名包含arm、step、port，其他权重归outputs/memory_vla/policy_servers/logs。核查A100两组各10个checkpoint，step001000..010000每1000一个，均config.json+两safetensors分片；最佳对照2000/M1 3000路径不变，4090不镜像完整权重。README已替换旧模式说明。验证bash -n/help、A100手选对照2000/M1 3000/M1 10000及GPU3/5571 dry-run、必填参数拒绝、5556已占端口拒绝、远端旧入口不存在均通过。未实际启动新模型/评估、未停服务、未改模型/训练；旧5555/5556 PID仍不变。下方--model预设记录仅为被取代的上一版交付历史。

最新交付（2026-11-10）：用户要求简化A100服务启动，新增公共scripts/run_groot_server.sh（本机及A100同路径，已可执行），--model official/official-native/baseline/m1，自动端口5556/5555/5562/5561、GPU0/0/2/1。训练后两组从--run-dir下selected_checkpoints.json读取最佳权重，默认本轮；源码默认resume_r4，环境n1d6-p0。支持--gpu-index/--port/--model-path/--source-dir/--python-path/--foreground/--dry-run，默认独立后台进程、日志统一归本轮logs并打印tail命令。端口占用直接拒绝，不kill/替换任何服务。README已将上一窗口长命令替换为简短入口用法，仅继续维护两份md。验证：bash -n、本机四配置dry-run，A100四配置dry-run及GPU/端口覆盖、真实已占5556端口启动拒绝均通过；5555/5556原PID不变，5561/5562仍未启动。此次仅实现/部署脚本，没有启动模型或评估，也未改网络/训练/队列。闭环中断状态仍未恢复。

本窗口补充（2026-11-10）：已核对旧服务真实cmdline/cwd/stdout：5555为原生接口、5556加sim wrapper，均官方权重、GPU0，启动时间分别11月6日09:23/11:31；B1K默认连5556。不能从现存进程确认历史保留原因。README新增四个端口的完整手动启动命令（原5555/5556及最佳对照5562、M1 5561）、环境与日志查询。此次未启动、停止或重启服务，只更新现有两份文档。

最新核查（2026-11-10 09:27本机/09:28 A100）：本轮闭环队列已失败退出，并非两组同时评估。baseline.log末尾FINISHED ok=False/statuses={'1':'interrupted'}，公共gpu1.log新增09:26:59 QUEUE_SIGNAL signal=15、09:27:01 QUEUE_INTERRUPTED；不能据此确认是谁发送信号，也不把旧01:xx仿真日志当成本轮进度。evaluation/status.json为failed，selected_policies仅baseline step2000、returncode=1。队列PID1895348及专用policy PID1921239均已不在，5562/5561均未监听，M1尚未启动；原5555/PID4055792、5556/PID905468仍监听。脚本实际为baseline GPU2/5562完成后才启动M1 GPU1/5561，顺序执行。此次仅查询与更新两份文档，未改代码、重启或停止任何进程；训练/独立验证完成结论不变。

最新状态（2026-11-10 09:12，只读核查）：两组训练均completed/10000步；M1训练状态文件05:06、对照06:11，两个验证器均completed，1000至10000共10个检查点全部验证。根status于09:04完成，selected_checkpoints已生成：M1 step3000，独立20Episode Loss=0.0027759973953851826；对照step2000，Loss=0.002761313837660035。M1最佳Loss略高，不能宣称记忆收益。M1选定点动作面板normal/empty/wrong MAE=0.0283838754/0.0089410975/0.0089148999，normal明显较差，仍需核查指标路径和逐Episode结果，不能仅据此下机制结论；本窗口仅查询状态，未修改训练/评估代码或进程。曲线已生成并镜像到本地plots/training_curves.png。闭环队列PID1895348仍在，状态closed_loop_queued_or_running，当前为对照step2000/端口5562/远端policy PID1921239；M1尚无selected_policy记录，闭环尚无本轮完成结论。公共gpu1.log包含旧时戳，不把其中旧记录当作本轮进度。原服务PID4055792/905468仍在。下方22:32记录为此前恢复经过。

最后更新：2026-11-09 22:32（北京时间）。第1000步停顿已定位并修复，两组均从完整step1000恢复，模型结构/预算不变。22:32:15同一状态快照显示M1训练1102、对照1089，同时独立验证器计算checkpoint1000；M1首Episode完成100/117窗口，对照已完成首Episode、进入第二个210260。确认训练与验证持续并行，不等验证结果。原Baseline服务未改。仅维护handoff.md与README.md。

**根因及修复：** 新目录validation_queue从未创建，atomic_json写step-001000.partial.json报FileNotFoundError。main捕获后写status-failed-rank-000.json，却在finally/destroy_process_group等待仍在训练的rank1，故traceback未及时输出，表现为rank0睡眠、rank1 GPU100%、主status仍training。上窗口将阻塞重点放在第二个barrier的推断不完整，并非已证明NCCL本身故障。两组真实失败状态现在归各arm/diagnostics/queue_failure_rank0.json。目录缺失回归测试先真实失败，再补atomic_json父目录创建通过；异常现在先输出完整traceback并失败退出，只在正常完成时集体销毁。保存文件协调另用Gloo CPU组、只一次完成屏障；NCCL绑定LOCAL_RANK设备，训练仍用NCCL。新增--resume-step严格匹配原world_size/seed/全局批量/冻结设置/数据/预算，加载HF权重、AdamW、各rank游标和detach bank，重新发布完整1000检查点；没有从零重跑或空记忆续接。r3初次恢复因未修目录也失败于第1001步之前；随后r4落实根因修复才成功，保留同一run与日志。

**查看日志：** root logs/*_torchrun.log每步输出JSON并flush。用户在logs目录用`tail -n 3 -F ../baseline/metrics/metrics.jsonl`看优化步/Loss，M1用../memory_v1/metrics/metrics.jsonl；metrics不混入警告，但会追加完整验证结果，注意kind。独立验证日志现在每Episode的首个窗口、每10窗及末尾输出validation_progress，记录checkpoint_step/episode/窗口计数。根status分列实际训练步和验证检查点，不能把validation checkpoint1000误读为训练停1000。

## 当前状态与下一步

用户最新要求暂停旧训练、重新开始、清理刚刚的输出、训练与验证并行，并追加让对照也用GPU0/1双卡。已实际落实：**M1 GPU2/3双卡、无记忆对照GPU0/1双卡同时训练**；M1验证GPU1，对照验证GPU0，各一个独立进程，等待每1000步固定检查点，不阻塞训练。原GPU0 Baseline服务5555/PID4055792、5556/PID905468未终止、未改权重/配置。用户明确允许利用GPU0剩余资源，服务共卡通过显式PID白名单实现。

当前run：`20261109_2100_task21_ddp_async_s42`，A100 `172.28.62.17`。控制器PID2894345，M1 torchrun2894394/ranks2894425、2894426，对照torchrun2894396/ranks2894413、2894414；M1验证2894395，对照验证2894397。本机镜像/闭环队列PID1895348。PIDs只作记录，操作前必须核对实际cmdline和status，不凭旧PID操作。

- 权威输出：`/nvme3/data_wyh/outputs/memory_vla/v1_groot_memoryvla/training/20261109_2100_task21_ddp_async_s42/`。
- 本机镜像：`outputs/v1_groot_memoryvla/training/20261109_2100_task21_ddp_async_s42/`，轻量产物30秒同步，不拉权重/optimizer。
- 当前两组训练、控制器和独立验证均使用：`/nvme3/projects/memory_vla/deployments/20261109_2100_task21_ddp_async_s42_resume_r4/`。不得覆盖运行中的可执行源码；README/handoff可同步。r1/r2为1000步前来源，r3为修队列之前的失败恢复来源；本机闭环队列仍引用r2的serve代码，其网络/Policy与r4一致，未改队列。
- Python：`/nvme3/projects/memory_vla/envs/n1d6-p0/bin/python`；数据：`/nvme3/data_wyh/datasets/b1k2025_groot_task21_v1_200/`；官方起点：`/nvme3/huggingface/train_model/GR00T/GR00T-N1.6-BEHAVIOR1k`。
- 两组固定180训练/20独立验证、seed42，不重新分组/拟合归一化。初始动作头SHA相同：`95f36ec58e298822e4eb5d97438621f1e0f38f071cd7e889a35b3e29253c7959`。Eagle BF16冻结/eval，动作专家FP32参数全训练，M1另训练原记忆；非LoRA。M1可训练1,527,984,416参数，对照1,418,624,000。AdamW1e-5/wd.01/clip1。
- 全局批量两组均为两条32决策窗/64决策，每rank一窗、micro4、stride8。虚拟槽固定Episode归属，条内顺时序，跨窗延续detach历史，不跨Episode或rank共享bank。两槽6630/6629窗，一遍13259窗需6630优化步，10000步约1.5遍；旧单窗run约0.75遍，不当作纯卡数速度对照。
- 每1000步保存HF模型/processor，rank0存最新optimizer，各rank存游标和detach bank；保存完成才发布validation_queue/step-*.json，训练继续。保存仍需短暂停顿拷贝写盘，验证计算不阻塞训练。当前已实测--resume-step 1000，入口脚本config/resume_remote.py，执行前须确认现有进程已停止，不重复运行。
- 每组独立验证器逐检查点覆盖20独立Episode全部完整窗口Loss，Episode等权；动作normal/empty/时间匹配wrong为每Episode固定中段32决策面板，前8/32步MAE及原始动作NPZ逐Episode保留。噪声seed不算Episode，面板尚未证实“历史必需”。
- best.json仅指向最低独立验证Loss的已保存step目录，无额外best全模型副本。两组同一验证点>=2000步且均连续4轮无0.1%改善时，共同约定当前最大优化步之后的1000步边界（多留一间隔）停止；验证滞后会推迟早停，最多10000。两组最终预算相同，不能独立停。训练结束后等待验证队列清空再生成selected_checkpoints，之后闭环。
- 两组每步日志都带step/total_steps=10000/progress_percent。训练：logs/memory_v1_torchrun.log、logs/baseline_torchrun.log；验证：logs/*_validation.log；完整梯度与耗时/吞吐在各组metrics/metrics.jsonl；根status分列train及validation checkpoint步。配对完整验证完成才生成plots/training_curves.png。
- 本轮1000步完整验证正在独立执行，尚无本轮best、正式收益或闭环结论；训练已越过1000。恢复后两组eval_interval/checkpoint_interval均为1000，原config/ddp.json保留原始启动参数，恢复实际命令见config/ddp_resume_step-001000.json与各arm/config/resume-step-001000.json。

**21:15仅对照切换经过：** 原控制器2678011、对照torchrun2678067/validator2678068已核对后结束，原单卡对照status记录125步，未保存任何checkpoint。其baseline目录和两份旧根日志现已永久删除，双卡对照从官方step0重新训练。M1 torchrun2678065与validator2678066原进程被新控制器接管，模型、optimizer、数据游标、跨窗记忆原样继续，未停止/重跑。真实M1在切换前200步、切换后240+步，对照新30+步正常。config/transition_2plus2.json记录准确原状态、保留PID及新设备；config/ddp_launch.json保留M1原命令，更新对照为0/1。start_remote_2plus2.py的--adopt-running-memory只用于此次接管，不可用于新run。

**本轮必要实测：** 官方GR00T+真实B1K双A100/NCCL完成4步，全部可训练参数两rank SHA逐值相同（692683f6fff30bbadbc08a95d2e7b7aff6e4ecd49db0e46f600e99d2cd861c08）。固定64决策/步，预热后单卡4.90秒、双卡2.66秒，短测1.84倍；每卡分配峰值35.5GiB。正式双卡对照约2.3秒/步、M1约2.6秒/步，异步验证共享计算后需看长跑日志。证据现统一在本轮diagnostics/preflight/task21_nccl_real与task21_nccl_single，均已镜像本机。两组实际config/ddp.json的动作头初始SHA、seed、Episode归属与全局批量一致，world_size均2。

本地全套 **25 passed**。新增队列目录缺失回归、恢复拒绝更改seed/批量，以及真实MemoryAdapter+AdamW断点恢复：下一32决策窗口Loss与全部更新参数逐值一致，两库合并从16延续到48，历史detach。已有Gloo两rank等价测试加入实际保存再继续两窗。A100真实M1从1000模型和AdamW/各rank银行开始训练，CPU协调保存约30.87秒后，两rank继续下一真实B1K窗口反传/优化，均保持2880条跨Episode独立记忆；日志及result.json在diagnostics/preflight/checkpoint_fixed_memory/。目录修复后正式两组已同时恢复并实际并行验证。远端未装pytest、未改依赖。旧4步train-only短测没有覆盖队列发布，此次补齐回归。新增临时探针权重共约62GiB已永久删除，仅保留这份小日志与JSON，不留额外训练run。

**显存共享已检查，但首次正式异步完整验证待观察：** 切换前真实对照训练运行时，同一探针进程在GPU1同时驻留两组验证模型并完成各32决策前向，Loss有限，探针峰值20.45GiB，全卡余22.61GiB。当前两验证器分散GPU0/1、各与一个对照rank共卡；GPU0另保留约14GiB原服务。新双卡对照每卡分配峰值33.1GiB，仍留有推理空间。探针与实际独立进程allocator不同，不将探针当作已完成1000步验证；观察真实峰值。config/validation_gpu_sharing_check.json可核查。

**旧run已永久清理（最新要求）：** 1725首轮、1755配对诊断、旧1930正式run及其本地/远端回收副本已删除；旧official_batch、acceptance、trainer_smoke权重、缓存、日志/曲线均删除。切换前单卡对照的回收目录/日志及当前run中已退出controller.log也删除。旧1930三个部署快照与DDP准备版快照已删除，本轮正在使用的两个快照保留。旧训练产物不再可从回收站恢复。共清理约134.4GiB旧产物。当前run与180/20划分、官方权重、数据、环境、现有评估和Baseline服务不变。本轮启动检查统一迁入diagnostics/preflight/{task21_ddp,task21_nccl_real,task21_nccl_single}；没有暂停任何当前训练/验证进程。outputs/index.json已删除旧训练入口，勿按旧路径找权重、恢复训练或尝试重建旧目录。

闭环队列：evaluation/scripts/evaluate_task21_trained.py，PID1895348，日志本轮logs/evaluation_queue.log，等本轮训练和验证完成才用selected_checkpoints。新专用5562/5561端口，Task21实例0–9、single1、seed0/sequence、action8、原生timeout，等待本机仿真卡空闲；产物归现有evaluation/experiment_1并以新run-tag追溯，不改原服务/台账。尚未执行闭环。

本轮代码集中task21_ddp.py、launch_task21_ddp.py、task21_async.py及test_task21_ddp.py，无网络结构修改；训练复用原formal的初始化/验证/错误历史实现。stop-that-shit用于约束只处理重启、多卡与必要并行机制。工作区无关改动仍属于用户，保留；官方参考目录不改，不另建Markdown。

## 基座、数据与环境

- GR00T 原版对照：`../references_code/Isaac-GR00T`，固定 `9b37aa1ce69c73c6d165233fa88128283bba4508`，未修改。
- 官方权重：`nvidia/GR00T-N1.6-BEHAVIOR1k`，revision `300db814db8ab5dd010026d5631f280048d06b91`，本地 `../../checkpoints/groot_n1d6_behavior1k/`。
- MemoryVLA 原始实现：`../references_code/memoryvla/vla/memory_vla.py`，SHA256 `998070d35cba0c57e7be85259c680aeee0120942545f04ad3bcd0bf5036d6f9a`。检索/融合/合并算子有直接对照原类的数值测试。
- 原始示教：`../../assets/datasets/b1k2025/` 的完整 Episode 210020，36,383 帧、30 Hz。转换结果 `../../assets/datasets/b1k2025_groot_v1_smoke/`，含三路 256×256 视频、82 维状态、23 维动作与完整 metadata。
- 转换逐行检查 256 维原始状态的冗余 sin/cos 和单位四元数；不能套用旧仿真示例中的 258 维索引。三路视频转换前后帧数、fps、起始时间一致。原始数据未覆盖。
- 独立 `.venv`：Python 3.10 / Torch 2.7.1+cu128 / Transformers 4.51.3。单 GPU 不装 DeepSpeed；训练用 FP32 参数 + BF16 autocast，冻结的 Eagle 为 BF16。原动作头 Beta 采样固定 CPU FP32。
- 环境复用既有模型环境的依赖文件，排除了 DeepSpeed、原 gr00t editable 和仿真包；必须 `PYTHONNOUSERSITE=1`，避免用户目录的 Torch/Transformers 覆盖。原有环境未修改。
- 核心源码最初复制完整官方 `gr00t/` 与 LICENSE，展开后改内部导入为 `v1_groot_memoryvla.*`；训练/服务入口为 `train.py` / `serve.py`，原 eval 移到根 `evaluation/` 共用。上游配置原样留在 `env/upstream.*`，不是当前项目的可直接执行 workspace 锁。

## 接入前证据与已否决方案

完整输入证据保留在 `audit/artifacts/input_audit.json`、`tokens_{eval,train,padded}.tsv` 和 `reference_memory_checks.json`，探针为 `audit/trace_b1k_inputs.py` 与 `audit/check_reference_memory.py`。早期 Eagle 形状探针使用随机权重，其结果不能冒充官方模型训练；后续官方权重验收见下方产物。

真实模板为 system + 单个 user，user 内容先语言后三图，无 assistant generation prompt；语言先转小写并去标点，collator 左 padding。EOS=151645、pad=151643、image=151669。长样本有效长度 323；短指令有效长度 282，左补 41 项。认知位置取最后有效 user EOS，不能使用 `attention_mask.sum()-1` 或固定索引。当前单 user 模板可以这样取；若未来改为多轮需保留 user span。

先前建议将感知 bank 扩为 2048 维并直接覆盖 LLM 后图像 Token，已被用户后续要求否决。当前唯一方案是 256 维感知和增量残差回注。也已更正“在线与示教完整指令相同”的错误：两者内容不同，见风险第 1 项。

避免使用子任务语言的提前注入分支；不要把有写库副作用的记忆函数放进 activation checkpoint 重算区。旧 16 项训练组不会接触合并后的历史分布；本轮已改为 32 次序列，读取/梯度/干预都覆盖合并后的记忆，但不等于已证明长程收益。

## 最终机制

固定感知 256 维、认知 2048 维、双库容量各 16、两层单头交叉注意力、时间编码加入历史 K、GateFusion、融合后写回、相邻最大相似度平均合并。未增加本体 Query、辅助损失或额外融合网络。

对当前视觉特征逐相机运行共享 `BottleneckSE(2048,512,256)`。不能将三相机的 243 个 Token 当作一个平方网格；每相机保留 81=9×9 布局，再按 processor 的 head/left/right 顺序拼回。

设原 LLM 后上下文特征为 H，SE 当前感知为 P，记忆融合为 P′。仅向原图像位置加入 `Linear256→2048(P′−P)`；该线性层无 bias，属于接口投影。没有使用 LLM 前的视觉 embedding 直接覆盖 H。认知取最后一个有效 user `<|im_end|>`，向该位置加入 `C′−C`。未修改其他位置、序列长度或两类 mask，AlternateVLDiT 的图像/非图像交叉注意力沿用原路由。

空历史直接返回当前 P/C，避免 BF16 下 `Gate(x,x)` 的舍入，确保残差严格为零。残差投影正常初始化，不用全零输出层阻断第一步检索梯度。

双库顺序都是：读严格过去 → 两层检索 → Gate → 返回当前可求导融合结果 → detach/clone 写入融合结果 → 超容量才合并。历史不做跨时间反传；当前动作 loss 可训练所有记忆参数。

## 实际 Tensor Shape

下表为旧验收使用完整示教指令的 16 项真实训练组；本轮统一简称后序列长度为 285，微批次 B=2，完整序列为 32 次。语言长度随指令与 padding 改变。

| Tensor | Shape / 语义 |
| --- | --- |
| input_ids / attention_mask | `[16,323]` |
| pixel_values | 48 张 `[1,3,252,252]` 图像，按样本/相机展开 |
| Vision 输出 | 每张 18×18×1152；重排到 9×9×4608 后经 mlp1 |
| visual_features | `[16,3,81,2048]`，LLM 前 mlp1 输出 |
| 感知当前/融合特征 | `[16,243,256]`，实现逐时刻处理，每次 `[1,243,256]` |
| 认知当前/融合特征 | `[16,1,2048]`，实现逐时刻处理，每次 `[1,1,2048]` |
| backbone_features | `[16,323,2048]`，残差前后长度和有效位置不变 |
| image_mask / backbone_attention_mask | `[16,323]`，原 mask 原样传给动作头 |
| state | `[16,1,128]`，82 个有效状态维度 |
| action / action_mask | `[16,50,128]`，每项 32×23=736 个有效标签元素 |
| 在线输出 | 两环境的 6 组动作分别 `[2,32,{3,4,7,1,7,1}]` |

第 0 帧完整指令下 user EOS 在位置 321，末尾换行为 322；图像位置为 63–143、151–231、239–319。实现按 input_ids 与有效 mask 动态查找，不把这些绝对位置写死。完整 Token 证据仍在 `audit/artifacts/tokens_*.tsv`。

## 历史采样与生命周期

`HistoryDataset` 按传入长度构建同 Episode 完整序列。新的训练 CLI 默认 `memory_train_sequence_length=32`：首组 `0,8,...,248`，次组从 256 开始，Episode 210020 生成 142 组。旧配置和 Dataset 默认 16 保留兼容（284 组），新 CLI 明确传入 32。所有动作标签需有完整 32 帧，不用重复末帧补组；未来 action 只作为监督目标，当前视觉、状态、语言全部 delta=0。

Collator 展开完整组并携带 Episode key、组号、frame_index、decision timestep、`sequence_length`。Sampler / worker / DDP 分发的单位是 Dataset item，因此不会拆组。Adapter 的 group 模式使用 context 的序列长度，而不再按旧 `group_size=16` 每 16 项清库。

首轮入口 `experiment/first_training.py` 使用 `mode=train_sequence` 与显式 `sequence_start/end`：同一序列按 2 决策分块，每块 backward，融合历史已 detach；32 决策完成后才 optimizer.step。开始/结束清库，中间块继承 bank；没有开始标记的续块会报错。官方冻结 Eagle 的真实输出先缓存到 CPU，之后将冻结 backbone 放 CPU，原动作头与全部记忆模块保持 FP32 参数、BF16 autocast 训练。无图像增强，不可把缓存训练速度当作完整端到端速度。

训练 timestep=`frame_index//8`，在线每执行一次决策增加 1，均对应 8 环境步。训练/在线第 17 次先读 16 条再写入，超过容量后独立在两库合并相邻平均 Token 余弦相似度最高的一对，保留较早时间戳。第 18 次开始读取已合并的 bank；32 次后每库 16 条、累计 16 次合并，合并后读取 15 次。

在线键为 `(session_id, env_id, episode_id)`。部分 Reset 按 session/env 找到旧键清除，不依赖 Reset 请求已经换成新 Episode ID。换 Episode ID 的首次请求也会清掉该环境旧库；其他环境和客户端保留。四次扩散去噪共享同一份已融合特征，不重复写记忆。

## 源码位置

下表保留原接入验收的定位；部分行号会随后续改动变化，优先用函数名查找。首轮新增入口和当前行号见后面的训练记录。

| 文件 / 函数 | 位置 | 修改目的 |
| --- | --- | --- |
| `data/convert_b1k.py` / `validate_raw_state`, `convert` | 45、92 | 真实 R1Pro 状态布局、三路视频和训练 metadata 转换 |
| `data/dataset/history_dataset.py` / `HistoryDataset.__getitem__` | 58 | 同 Episode 当前观测与未来动作标签，完整时序组 |
| 同上 / `HistoryCollator.__call__` | 84 | 组展开、顺序/间隔检查、记忆上下文 |
| `utils/video_utils.py` / `get_frames_by_indices` | 369 | PyAV 单次 seek 索引读取组；缺帧报错 |
| `model/modules/nvidia/Eagle-Block2A-2B-v2/modeling_eagle3_vl.py` / `forward` | 213 | 可选返回本次 mlp1 特征，不重复视觉编码 |
| `model/modules/eagle_backbone.py` / `forward` | 106 | 暴露 B1K 三相机网格，保留原 VLM 输出 |
| `model/memory/bank.py` / `MemoryBank.read_fuse_write`, `_write` | 106、128 | 原检索、时间编码、Gate、融合写入及相邻合并 |
| `model/memory/adapter.py` / `MemoryAdapter.forward` | 29 | SE、动态 user EOS、增量残差、组边界 |
| `model/gr00t_n1d6/gr00t_n1d6.py` / `_init_weights` | 480 | 官方基座缺失的新增 memory 参数正确初始化 |
| 同上 / `prepare_input`, `_backbone_with_memory` | 490、523 | 元数据不混入 Tensor 树；训练与推理统一接入 |
| 同上 / `forward`, `get_action` | 532、551 | 同一记忆路径后调用原动作头 |
| `configs/model/gr00t_n1d6.py` | `memory_*` 字段 | 参数与配置可完整 save_pretrained 重载 |
| `model/gr00t_n1d6/setup.py` | `_create_model`, `_create_dataset`, `_create_collator` | 记忆开关、完整权重检查、历史 loader 接通 |
| `experiment/launch_finetune.py` | CLI 构建 | 从官方模型配置继承 horizon/维度；帧 batch 转完整组 |
| `experiment/trainer.py` / `get_train_dataloader` | 206 | 历史 Dataset 使用组级 map-style sampler |
| `experiment/experiment.py` | 最终保存 | output 根目录同时保存 processor，可直接被 Policy 加载 |
| `policy/memory_state.py` | `context`, `advance`, `reset` | 在线隔离、决策时钟与部分 Reset |
| `policy/gr00t_policy.py` / `_get_action`, `reset` | 308、415 | options 接通模型上下文及真实 Policy Reset |

## 验收结果

| 验收 | 结果与证据 |
| --- | --- |
| 官方 checkpoint + 真实单批次 | 无 missing/unexpected keys；loss=0.0010521539，梯度范数=1.02210，AdamW step=1，vlln 参数实际变化 |
| 无历史输出一致性 | 相同官方权重、相同扩散随机数，完整动作预测逐值相等 |
| 有历史梯度传播 | 真实 16 项组 loss=0.01200632，所有 memory 参数梯度存在且有限；SE、双路 Q、时间编码、Gate、残差投影均非零，优化后残差权重变化 |
| 训练/推理时序 | 相同真实 Eagle 特征下，16 项训练组与 16 次在线读/融合/写回逐值相等；独立单测还检查未来观测扰动不影响过去 |
| 第 17 次写入 | 两库各保持 16 项，各记录 1 次合并；单测验证选相邻最大相似对、均值及保留较早时间 |
| 多环境独立 Reset | 双环境库独立；部分 Reset 后另一环境条目保留；另有独立 session 隔离单测 |
| 模型保存/重载 | 完整 HF checkpoint 无缺失/多余参数；配置恢复，runtime bank 为空；重新建立相同历史后冷启动及有历史动作均逐值相等 |
| 正式训练 CLI | 原始官方基座 → Dataset/Collator → Trainer → 真实一步训练与保存，train_loss=0.0075876601 |
| 正式 BF16 Policy | 加载 Trainer 最终 output，双环境 get_action、每决策单次写入、部分 Reset、全 Reset 通过 |
| 原方法一致性 | 8 个机制/配置测试通过，包括直接加载未修改 MemoryVLA 原类，比较 18 次检索/融合/合并输出及时间戳；配置默认值显式写入 checkpoint |
| 版本测试全集 | `pytest src/v1_groot_memoryvla/tests -q`：10 passed，含 2 项包结构/资源检查 |
| 视频读取 | PyAV 与官方 ffmpeg 在帧 0、8、25112 的 RGB 逐像素一致，最大差为 0 |

上述为历史验收结论；原验收权重与日志已按最新清理要求永久删除，不再提供文件链接。本轮启动证据在当前run的diagnostics/preflight/，最新代码测试22 passed。

复现命令见 [README](README.md)。GPU 验收峰值约 27.5 GiB，耗时约 39 秒；共享 GPU 上其他任务导致的显存不足不作为算法失败，成功报告来自空闲 GPU 的完整运行。

## 尚需确认与实验边界

1. **语言已统一。** Processor 从数据 metadata 建立完整指令/任务名/仿真简称 → 公共任务简称的映射，随 `processor_config.json` 保存。210020 两种实际文本的 input_ids 和 attention mask 完全一致，重载映射通过；原始完整示教不改。已有旧 checkpoint 没有此映射，不要误当成统一后的模型。
2. **旧诊断为8训练/4验证Episode；本轮正式训练为180/20、单seed42。** 全部同一场景模板，不能声称跨场景泛化。官方基座已后训练50任务，其Episode清单不可得；只保证本轮微调隔离。
3. **时序一致性的严格比较隔离了 VLM 数值差异。** CUDA BF16 的 batch-size 变化可能改变 Eagle 舍入。比较使用相同真实 Eagle 输出，只核记忆读取/写入的因果顺序与一致性；没有把训练 dropout 与推理输出逐值相等作为要求。SE 按单时刻三相机运行，保证两种场景采用相同卷积 batch。
4. **本轮两组各双GPU DDP，未使用ZeRO。** 已从各自1000步恢复HF模型、AdamW、每rank游标与银行；CPU下一窗口等价及A100真实继续均通过。当前恢复限定同world_size/虚拟槽/数据顺序与超参，不支持换卡数后接续旧bank；不能只加载权重后从中间窗口空库继续。
5. **支持当前 B1K 三相机各 9×9 网格。** 其他视图数、分辨率或非平方网格明确报错。旧convert_b1k聚合统计占用较大；本轮prepare_task21_formal逐Episode并行转换、直接沿用官方统计，已完成全部200条，不重新拟合验证数据。
6. **正式训练和闭环结果尚未产生。** 原仿真/基线服务与评分协议未切换。本轮按新指示启动正式180/20配对实验及完成后的闭环队列，旧诊断无一致历史收益不能写成正式结论。

## 历史结果摘要

早期40步训练和8训练/4验证Episode的200步诊断均未证实一致的正确历史收益；不要把动作变化或训练Loss下降当成该结论。对应权重、缓存、逐步日志、图表和旧报告已按用户要求永久删除，后续以本轮180/20独立Episode配对训练与闭环结果为准。模型、数据转换与训练/验收源码保留，未因清理而重做诊断。
最新执行（2026-11-11 10:35）：用户要求训练启动页能取消GPU等待/任务排队。已实现训练详情“权重与任务控制”中的取消按钮：pending joint队列显示全部affected_runs；waiting_for_four_runs显示“取消本组排队”，waiting_for_gpus显示“取消 GPU 等待及本组排队”。当前联合Baseline+Memory共用一个控制器，取消范围是这两组一起，页面明确列出，不支持假装只取消一组。运行中的Task40只允许原有未来1000整数倍保存后停止；queued/cancelled/completed禁用停止点。后台数据准备队列独立，取消训练不会取消数据准备。本轮没有实际撤销任何真实队列、训练、服务或评估。

实现：train/dashboard_remote.py新增WAITING_QUEUE_STATES、queue_pending、terminate_waiting_controller和cancel；jobs带can_cancel/can_stop及queue.affected_runs；train/dashboard.py透传cancel，evaluation/scripts/view_results.py新增POST /api/training/cancel；train/dashboard.html/js加取消状态/范围提示和按钮。取消时队列flock，验证所有arm仍queued/step0且无web_process；pidfd绑定process.json中PID，核对同账号、完整argv、deployments内joint_task_queue.py及精确config路径，再SIGSTOP精确控制器、确认暂停、重查队列/各arm且无子进程才SIGTERM，finally SIGCONT（拒绝时恢复）。不杀进程组，不触碰冻结训练源码，不终止其他GPU进程。取消后arm/status及queue/status标cancelled，queue/cancellation.json保留审计，日志/配置/源码不删除；重复取消幂等，联合取消后需重新安排。

验收：42项tests/test_groot_training_dashboard.py、test_training_dataset_inventory.py、test_groot_inference_launch.py通过；包含本地真实临时队列控制器与无关进程、启动竞态拒绝后恢复、错误/复用PID拒绝、配对范围与HTTP路由；JS node --check和Python编译通过。8765已SIGINT温和重载3292212→3349026，恢复原评估3267977且/proc startticks170952560不变；现网API实际显示联合两组can_cancel=true/can_stop=false、Task40两组can_cancel=false/can_stop=true。当前联合仍waiting_for_four_runs，Task40继续training，没有调用真实cancel接口。其他账号8766/PID3276319未动。README已说明操作及范围，仅维护这两份md。
