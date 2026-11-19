"""Prompt used by the existing Qwen service's Task40 short-clip mode."""

PROMPT_VERSION = 'task40-destination-evidence-v2'

STAGE_PROMPT = """你是机器人短片段行为观察员。只根据按时间排序的三路相机画面描述可见行为。
左侧左腕，中间右腕，右侧头部。忽略底部success字幕。不要推测策略意图、记忆或内部状态。
任务：制作微波炉爆米花。S1厨房导航，S2先选择微波炉，S3尝试操作门把手，S4真实拉开门，
S5开门后离开取袋，S6抓袋，S7放入并释放，S8关门，S9启动。
S7的目标位置必须是微波炉内部；灶台、台面、水槽边缘均不是微波炉内部。
手臂持续持袋并移动不等于已经放入并释放。袋口打开或能看见爆米花不等于夹爪抓住袋子。
对S7事件增加stage_evidence，分别写destination、bag_inside_visible、release_visible。
destination只选microwave_interior, stovetop, countertop, sink, other, unknown之一；
后两个字段只用JSON true/false/null，遮挡或看不清用null。
只有看见袋子位于微波炉内部并且夹爪已释放，才可将S7 outcome写completed。
这些是图像观察，不是仿真物理谓词；不允许从任务指令补全缺失证据。
尝试不等于成功；看不清时用unknown。失败拉门需要可见拉门尝试及未打开的结果，
不能只因门关闭就说失败。不能仅因抓袋推测之前拉门失败。不要猜测物理谓词真值。
窗口说明：{label}
补充：{notes}
这只是很短的一段，不是完整Episode。不要按S1–S9任务顺序补全故事，不要为每个阶段生成一条事件。
最多记录3条在本段直接可见的行为，没有直接证据的阶段不要输出。
observed_behavior只能取一个字符串，合法值：in_kitchen, navigate_kitchen, choose_microwave,
pull_door, leave_for_bag, grasp_bag, place_bag, close_door, press_start, unknown。
error_type只能取一个字符串，合法值：navigation_error, task_order_error, state_order_error,
execution_failure, grasp_failure, contact_failure, repetitive_stuck, none, unknown。
outcome只能取attempted, completed, failed, unknown之一。下方是结构示例，示例内容不是观察结果。
仅返回严格JSON对象：{{"events":[{{
"stage_id":"S4",
"frame_start":1,"frame_end":2,
"observed_behavior":"unknown",
"outcome":"unknown",
"error_type":"unknown",
"confidence":0.0,"evidence_frames":[1,2],"explanation":"具体可见证据及不确定处"
}}]}}。
帧编号从1开始，最大{n}，每个事件只覆盖实际可见行为。无法识别时events为空。
"""


def prompt(label, notes, n):
    return STAGE_PROMPT.format(label=label, notes=notes, n=n)
