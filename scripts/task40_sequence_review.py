"""Vocabulary and renderer for chronological Task40 video annotation."""
import json
from pathlib import Path


MOVE_STATES = [
    ("arrived", "成功到达"), ("arrived_after_retries", "来回调整后到达"),
    ("not_arrived", "尝试前往，但没有到达"), ("not_arrived_after_retries", "反复移动/调整，仍没有到达"),
    ("wrong_destination", "去了其他位置"),
    ("in_progress", "仍在移动，视频还没显示结果"), ("already_there", "这段开始时已经在这里"),
]
OPERATE_STATES = [
    ("success_once", "尝试一次就成功"), ("success_after_retries", "反复尝试后成功"),
    ("success_count_unknown", "成功了，尝试次数看不全"),
    ("failed_once", "尝试一次，未成功"), ("failed_repeatedly", "一直/反复尝试，仍未成功"),
    ("failed_count_unknown", "未成功，尝试次数看不全"),
    ("in_progress", "仍在尝试，视频还没显示结果"),
]
UNKNOWN = [("unknown", "看不清/无法判断")]


def action(key, label, states, details=()):
    return {"id": key, "label": label, "states": states + UNKNOWN,
            "details": [("unspecified", "不补充/看不清")] + list(details)}


ACTIONS = [
    action("go_kitchen", "前往/到达厨房", MOVE_STATES),
    action("go_bag", "前往/返回袋子旁", MOVE_STATES),
    action("go_microwave", "前往/返回微波炉旁", MOVE_STATES),
    action("open_door", "尝试打开微波炉门", OPERATE_STATES + [
        ("partly_open", "只打开了一部分"), ("already_open", "这段开始时门已经开着")], [
        ("missed_handle", "没碰到/没抓住把手"), ("contact_no_open", "有接触/拉动，但门没打开"),
        ("visible_obstacle", "画面可见物体挡住门")]),
    action("grasp_bag", "抓取/拿起袋子", OPERATE_STATES + [
        ("dropped_after_grasp", "拿起来后又掉了"), ("already_held", "这段开始时已经拿着袋子")], [
        ("missed_bag", "夹爪没碰到/没夹住袋子"), ("not_lifted", "碰到袋子但没拿起"),
        ("door_closed", "此时微波炉门仍关着"), ("door_open", "此时微波炉门已打开")]),
    action("carry_bag", "拿着袋子移动/返回微波炉", MOVE_STATES + [
        ("dropped", "移动时袋子掉了")]),
    action("place_bag", "尝试把袋子放入微波炉", OPERATE_STATES + [
        ("inside_still_held", "袋子进了炉腔，但还没松手"),
        ("outside", "松手/掉下后，袋子在炉外")], [
        ("closed_door", "门关着，袋子被门挡住"), ("partly_open_door", "门没完全打开，袋子受阻"),
        ("missed_opening", "门已开，但没对准炉腔/碰到门框"),
        ("other_obstacle", "被其他可见物体挡住"), ("still_held", "尝试后袋子仍拿在手里"),
        ("empty_hand", "没拿着袋子，却在做放入动作")]),
    action("release_elsewhere", "在炉外放下/松开袋子", [
        ("placed", "袋子放在台面/其他位置"), ("fell", "袋子掉到地面"),
        ("still_held", "做了松手动作，但袋子还拿着"), ("in_progress", "还没显示结果")]),
    action("drop_bag", "袋子滑落/掉落", [
        ("outside", "掉在炉外/地面"), ("inside", "落在炉内"), ("caught_again", "掉落后又接住")]),
    action("close_door", "尝试关闭微波炉门", OPERATE_STATES + [
        ("partly_closed", "只关上一部分"), ("already_closed", "这段开始时门已经关着")], [
        ("bag_blocks", "袋子/其他可见物体挡住门"), ("bag_outside", "袋子还没放进炉内"),
        ("missed_door", "没有碰到门/推到有效位置")]),
    action("start_microwave", "尝试启动微波炉", OPERATE_STATES + [
        ("already_on", "这段开始时已经启动")], [
        ("door_open", "门仍开着"), ("bag_outside", "袋子仍在炉外"),
        ("no_visible_response", "操作后未见启动指示/响应"), ("missed_control", "没有碰到启动控件")]),
    action("wait", "停住/等待/无明显操作", [
        ("stationary", "基本不动"), ("small_motion", "只有小幅调整"),
        ("unrelated_repeat", "持续重复无明显进展的动作")]),
    action("other", "其他可见动作（用一句话补充）", [("observed", "看到了这个动作")]),
    action("unclear", "看不出在做什么", []),
]


def build_sequence_page(segments):
    data = {"actions": ACTIONS, "videos": segments}
    encoded = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    return Path(__file__).with_name("task40_sequence_review.html").read_text(encoding="utf-8").replace("__REVIEW_DATA__", encoded)
