import json
import re
import time
from typing import Optional

import numpy as np
from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from utils import logger


PAUSE_TEMPLATE = "暂停.png"
EXIT_TEMPLATE = "退出战斗.png"
EXIT_CONFIRM_TEMPLATE = "退出战斗确认.png"
BATTLE_END_TEMPLATE = "战斗结束.png"
BATTLE_END_CONFIRM_TEMPLATE = "战斗结束_确定.png"
BATTLE_FAILED_TEMPLATE = "战斗失败.png"
BATTLE_FAILED_CONFIRM_TEMPLATE = "战斗失败_确定.png"
BLOODLINE_SELECTION_TEMPLATE = "血缘选择标志.png"
WAITING_SELECTION_TEMPLATE = "等待选择.png"
FINAL_BOSS_TEMPLATE = "最终boss.png"
SOUL_DUNGEON_TEMPLATE = "魂系副本.png"
ENDLESS_TAB_TEMPLATE = "无尽深渊tab.png"
BLOODLINE_TAB_TEMPLATE = "血缘.png"

RIGHT_SOUL_ROI = [656, 174, 251, 325]
LEFT_SOUL_ROI = [367, 174, 262, 328]

# 跟 pipeline 里的点击区域保持一致。
CHOOSE_LEFT_WHEN_RIGHT_SOUL = [434, 190, 124, 62]
CHOOSE_RIGHT_WHEN_LEFT_SOUL = [696, 187, 163, 75]
DEFAULT_LEFT_CHOICE = [434, 187, 122, 68]
MAYBE_BLESSING_CHOICE = [315, 216, 75, 87]
EXIT_CONFIRM_TARGET = [511, 472, 77, 26]
BATTLE_FAILED_CONFIRM_TARGET = [1117, 669, 86, 33]
BATTLE_FAILED_CONFIRM_POINTS = [
    (1164, 683),
    (1160, 685),
    (1140, 683),
    (1190, 683),
    (1164, 674),
    (1164, 696),
]
PAUSE_FALLBACK_POINTS = [
    (21, 26),
    (26, 24),
    (31, 27),
    (18, 31),
]
BLOODLINE_LEVEL_OCR_ROI = [868, 236, 230, 241]
BLOODLINE_LEVEL_KEYWORDS = ["普通", "困难", "噩梦", "地狱", "疯狂"]


def _safe_screencap(context: Context) -> Optional[np.ndarray]:
    image = context.tasker.controller.post_screencap().wait().get()
    if image is None or not isinstance(image, np.ndarray):
        logger.info("退出当前血缘战斗: 截图失败")
        return None
    return image


def _match_template(
    context: Context,
    image: np.ndarray,
    template: str,
    *,
    threshold: float = 0.9,
    roi: Optional[list[int]] = None,
) -> bool:
    node = "BloodlineExitTemplate"
    config = {
        "recognition": "TemplateMatch",
        "template": template,
        "threshold": threshold,
    }
    if roi is not None:
        config["roi"] = roi

    reco_detail = context.run_recognition(
        node,
        image,
        pipeline_override={node: config},
    )
    return bool(reco_detail and reco_detail.hit)


def _click_point(context: Context, x: int, y: int, label: str) -> None:
    logger.info(f"退出当前血缘战斗: 点击{label} ({x}, {y})")
    context.tasker.controller.post_click(x, y).wait()


def _click_rect_center(context: Context, rect: list[int], label: str) -> None:
    x, y, w, h = rect
    click_x = x + w // 2
    click_y = y + h // 2
    _click_point(context, click_x, click_y, label)


def _sleep_after_battle_failed_click(settle_sec: float, attempt_no: int) -> None:
    if attempt_no >= 6:
        delay_sec = max(settle_sec, 1.4)
    elif attempt_no >= 3:
        delay_sec = max(settle_sec, 1.0)
    else:
        delay_sec = settle_sec

    if delay_sec > 0:
        time.sleep(delay_sec)


def _click_battle_failed_confirm(
    context: Context,
    image: np.ndarray,
    *,
    attempt_no: int,
    threshold: float = 0.8,
    settle_sec: float = 0.6,
    prefer_template: bool = True,
) -> bool:
    if prefer_template and (attempt_no == 1 or attempt_no % 4 == 0):
        if _click_template_center(
            context,
            image,
            BATTLE_FAILED_CONFIRM_TEMPLATE,
            threshold=threshold,
            settle_sec=0,
        ):
            _sleep_after_battle_failed_click(settle_sec, attempt_no)
            return True

    point_x, point_y = BATTLE_FAILED_CONFIRM_POINTS[
        (attempt_no - 1) % len(BATTLE_FAILED_CONFIRM_POINTS)
    ]
    _click_point(context, point_x, point_y, f"战斗失败确定兜底第 {attempt_no} 次")
    _sleep_after_battle_failed_click(settle_sec, attempt_no)
    return True


def _click_template_center(
    context: Context,
    image: np.ndarray,
    template: str,
    *,
    threshold: float = 0.9,
    settle_sec: float = 0.2,
) -> bool:
    node = "BloodlineExitClickTemplate"
    reco_detail = context.run_recognition(
        node,
        image,
        pipeline_override={
            node: {
                "recognition": "TemplateMatch",
                "template": template,
                "threshold": threshold,
            }
        },
    )
    if not (reco_detail and reco_detail.hit and reco_detail.box):
        return False

    x, y, w, h = reco_detail.box
    click_x = x + w // 2
    click_y = y + h // 2
    _click_point(context, click_x, click_y, f" {template}")
    if settle_sec > 0:
        time.sleep(settle_sec)
    return True


def _click_pause_for_exit(
    context: Context,
    image: np.ndarray,
    *,
    attempt_no: int,
    settle_sec: float,
) -> bool:
    prefer_fallback = attempt_no > 1 and attempt_no % 3 == 0
    if not prefer_fallback and _click_template_center(
        context,
        image,
        PAUSE_TEMPLATE,
        threshold=0.8,
        settle_sec=settle_sec,
    ):
        logger.info(f"退出当前血缘战斗: 点击暂停第 {attempt_no} 次（模板）")
        return True

    point_x, point_y = PAUSE_FALLBACK_POINTS[
        (attempt_no - 1) % len(PAUSE_FALLBACK_POINTS)
    ]
    _click_point(context, point_x, point_y, f"暂停兜底第 {attempt_no} 次")
    if settle_sec > 0:
        time.sleep(settle_sec)
    return True


def _extract_ocr_texts(
    context: Context,
    image: np.ndarray,
    roi: list[int],
    *,
    threshold: float = 0.3,
) -> list[str]:
    reco_detail = context.run_recognition(
        "BloodlineExitOCR",
        image,
        pipeline_override={
            "BloodlineExitOCR": {
                "recognition": "OCR",
                "roi": roi,
                "threshold": threshold,
            }
        },
    )
    if not (reco_detail and reco_detail.all_results):
        return []
    return [
        str(result.text).strip()
        for result in reco_detail.all_results
        if getattr(result, "text", None)
    ]


def _bloodline_stage_list_visible(context: Context, image: np.ndarray) -> bool:
    recognized_texts = _extract_ocr_texts(context, image, BLOODLINE_LEVEL_OCR_ROI)
    logger.info(f"切回血缘: 关卡区 OCR: {recognized_texts or '无'}")
    for text in recognized_texts:
        if any(keyword in text for keyword in BLOODLINE_LEVEL_KEYWORDS) and re.search(
            r"\d",
            text,
        ):
            return True
    return False


def _battle_end_visible(context: Context, image: np.ndarray) -> bool:
    return _match_template(context, image, BATTLE_END_TEMPLATE) and _match_template(
        context,
        image,
        BATTLE_END_CONFIRM_TEMPLATE,
    )


def _battle_failed_visible(context: Context, image: np.ndarray) -> bool:
    return _match_template(context, image, BATTLE_FAILED_TEMPLATE) and _match_template(
        context,
        image,
        BATTLE_FAILED_CONFIRM_TEMPLATE,
        threshold=0.8,
    )


def _battle_selection_visible(context: Context, image: np.ndarray) -> bool:
    return _bloodline_initial_selection_visible(
        context,
        image,
    ) or _waiting_selection_visible(context, image)


def _bloodline_initial_selection_visible(context: Context, image: np.ndarray) -> bool:
    return _match_template(context, image, BLOODLINE_SELECTION_TEMPLATE)


def _waiting_selection_visible(context: Context, image: np.ndarray) -> bool:
    return _match_template(context, image, WAITING_SELECTION_TEMPLATE)


def _click_next_selection(context: Context, image: np.ndarray) -> None:
    if _match_template(context, image, SOUL_DUNGEON_TEMPLATE, roi=RIGHT_SOUL_ROI):
        _click_rect_center(context, CHOOSE_LEFT_WHEN_RIGHT_SOUL, "右侧魂系对面的左侧选项")
    elif _match_template(context, image, SOUL_DUNGEON_TEMPLATE, roi=LEFT_SOUL_ROI):
        _click_rect_center(context, CHOOSE_RIGHT_WHEN_LEFT_SOUL, "左侧魂系对面的右侧选项")
    else:
        _click_rect_center(context, DEFAULT_LEFT_CHOICE, "默认左侧选项")

    time.sleep(0.2)
    _click_rect_center(context, MAYBE_BLESSING_CHOICE, "可能的赐福关")


def _click_selection_before_exit(
    context: Context,
    image: np.ndarray,
    *,
    settle_sec: float,
) -> bool:
    if _bloodline_initial_selection_visible(context, image):
        _click_rect_center(context, DEFAULT_LEFT_CHOICE, "非奶牛关左侧选项")
    elif _waiting_selection_visible(context, image):
        _click_next_selection(context, image)
    else:
        return False

    if settle_sec > 0:
        time.sleep(settle_sec)
    return True


def _parse_float(params: dict, key: str, default: float) -> float:
    raw = params.get(key, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.info(f"退出当前血缘战斗: {key} 参数非法 {raw!r}，使用默认值 {default}")
        return default


def _parse_int(params: dict, key: str, default: int) -> int:
    raw = params.get(key, default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.info(f"退出当前血缘战斗: {key} 参数非法 {raw!r}，使用默认值 {default}")
        return default


@AgentServer.custom_action("bloodline_exit_current_battle")
class BloodlineExitCurrentBattleAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        try:
            params = json.loads(argv.custom_action_param or "{}")
        except json.JSONDecodeError:
            logger.info("退出当前血缘战斗: custom_action_param 非法，使用默认参数")
            params = {}
        if not isinstance(params, dict):
            params = {}

        max_wait_sec = _parse_float(params, "max_wait_sec", 30.0)
        loop_sleep_sec = _parse_float(params, "loop_sleep_sec", 0.1)
        select_before_exit = bool(params.get("select_before_exit", True))
        selection_settle_sec = _parse_float(params, "selection_settle_sec", 1.0)
        selection_retry_interval_sec = _parse_float(
            params,
            "selection_retry_interval_sec",
            1.2,
        )
        pause_settle_sec = _parse_float(params, "pause_settle_sec", 0.7)
        pause_retry_interval_sec = _parse_float(
            params,
            "pause_retry_interval_sec",
            1.0,
        )
        exit_click_settle_sec = _parse_float(params, "exit_click_settle_sec", 0.6)
        started_at = time.monotonic()
        pause_clicked_count = 0
        exit_clicked_count = 0
        selection_clicked_count = 0
        last_selection_clicked_at = 0.0
        last_pause_clicked_at = 0.0

        while True:
            now = time.monotonic()
            elapsed_sec = now - started_at
            if max_wait_sec > 0 and elapsed_sec >= max_wait_sec:
                logger.info(
                    f"退出当前血缘战斗: 等待退出时机超时 {elapsed_sec:.1f}s，"
                    f"已点击选择 {selection_clicked_count} 次，"
                    f"已点击暂停 {pause_clicked_count} 次，"
                    f"已点击退出 {exit_clicked_count} 次"
                )
                return False

            image = _safe_screencap(context)
            if image is None:
                if loop_sleep_sec > 0:
                    time.sleep(loop_sleep_sec)
                continue

            if _battle_end_visible(context, image):
                logger.info("退出当前血缘战斗: 已到战斗结束弹窗")
                context.override_next(argv.node_name, ["血缘_战斗结束"])
                return True

            if _battle_failed_visible(context, image):
                logger.info("退出当前血缘战斗: 已到战斗失败弹窗")
                context.override_next(argv.node_name, ["战斗失败"])
                return True

            if _match_template(context, image, EXIT_CONFIRM_TEMPLATE):
                logger.info("退出当前血缘战斗: 已打开退出确认弹窗")
                context.override_next(argv.node_name, ["退出战斗确认"])
                return True

            if _match_template(context, image, EXIT_TEMPLATE):
                exit_clicked_count += 1
                if _click_template_center(
                    context,
                    image,
                    EXIT_TEMPLATE,
                    settle_sec=exit_click_settle_sec,
                ):
                    logger.info(
                        f"退出当前血缘战斗: 已打开暂停菜单并点击退出第 {exit_clicked_count} 次"
                    )
                    continue

            if select_before_exit and _battle_selection_visible(context, image):
                if (
                    selection_clicked_count > 0
                    and now - last_selection_clicked_at < selection_retry_interval_sec
                ):
                    if loop_sleep_sec > 0:
                        time.sleep(loop_sleep_sec)
                    continue

                selection_clicked_count += 1
                if _click_selection_before_exit(
                    context,
                    image,
                    settle_sec=selection_settle_sec,
                ):
                    logger.info(
                        f"退出当前血缘战斗: 选择后再退出第 {selection_clicked_count} 次"
                    )
                    last_selection_clicked_at = time.monotonic()
                    continue

            if (
                pause_clicked_count > 0
                and now - last_pause_clicked_at < pause_retry_interval_sec
            ):
                if loop_sleep_sec > 0:
                    time.sleep(loop_sleep_sec)
                continue

            if _battle_selection_visible(context, image):
                pause_clicked_count += 1
                logger.info("退出当前血缘战斗: 选择界面未启用预选择，尝试暂停等待菜单")
                _click_pause_for_exit(
                    context,
                    image,
                    attempt_no=pause_clicked_count,
                    settle_sec=pause_settle_sec,
                )
                last_pause_clicked_at = time.monotonic()
                continue

            if _match_template(context, image, PAUSE_TEMPLATE, threshold=0.8):
                pause_clicked_count += 1
                _click_pause_for_exit(
                    context,
                    image,
                    attempt_no=pause_clicked_count,
                    settle_sec=pause_settle_sec,
                )
                last_pause_clicked_at = time.monotonic()
                continue

            if _click_template_center(context, image, FINAL_BOSS_TEMPLATE):
                logger.info("退出当前血缘战斗: 已进入最终 boss，继续等待可退出状态")
                if loop_sleep_sec > 0:
                    time.sleep(loop_sleep_sec)
                continue

            if loop_sleep_sec > 0:
                time.sleep(loop_sleep_sec)


@AgentServer.custom_action("bloodline_handle_battle_failed")
class BloodlineHandleBattleFailedAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        try:
            params = json.loads(argv.custom_action_param or "{}")
        except json.JSONDecodeError:
            logger.info("处理战斗失败: custom_action_param 非法，使用默认参数")
            params = {}
        if not isinstance(params, dict):
            params = {}

        max_wait_sec = _parse_float(params, "max_wait_sec", 18.0)
        loop_sleep_sec = _parse_float(params, "loop_sleep_sec", 0.2)
        settle_sec = _parse_float(params, "settle_sec", 0.6)
        gone_required_hits = _parse_int(params, "gone_required_hits", 2)
        soft_success_after_clicks = _parse_int(params, "soft_success_after_clicks", 6)
        started_at = time.monotonic()
        clicked_count = 0
        gone_hits = 0

        while True:
            elapsed_sec = time.monotonic() - started_at
            if max_wait_sec > 0 and elapsed_sec >= max_wait_sec:
                logger.info(
                    f"处理战斗失败: 等待弹窗关闭超时 {elapsed_sec:.1f}s，"
                    f"已点击 {clicked_count} 次"
                )
                if clicked_count >= soft_success_after_clicks:
                    logger.info("处理战斗失败: 已多次尝试确认，继续交给切 tab 流程兜底")
                    return True
                return False

            image = _safe_screencap(context)
            if image is None:
                if loop_sleep_sec > 0:
                    time.sleep(loop_sleep_sec)
                continue

            failed_title_visible = _match_template(
                context,
                image,
                BATTLE_FAILED_TEMPLATE,
                threshold=0.8,
            )
            failed_confirm_visible = _match_template(
                context,
                image,
                BATTLE_FAILED_CONFIRM_TEMPLATE,
                threshold=0.8,
            )

            if failed_title_visible or failed_confirm_visible:
                gone_hits = 0
                next_attempt_no = clicked_count + 1
                if _click_battle_failed_confirm(
                    context,
                    image,
                    attempt_no=next_attempt_no,
                    threshold=0.8,
                    settle_sec=settle_sec,
                    prefer_template=failed_confirm_visible,
                ):
                    clicked_count = next_attempt_no
                    logger.info(f"处理战斗失败: 点击确定第 {clicked_count} 次")
                    continue

            gone_hits += 1
            if clicked_count == 0:
                logger.info("处理战斗失败: 未再看到战斗失败弹窗，视为已处理")
                return True
            if gone_hits >= gone_required_hits:
                logger.info("处理战斗失败: 弹窗已连续消失，处理完成")
                return True

            if loop_sleep_sec > 0:
                time.sleep(loop_sleep_sec)


@AgentServer.custom_action("bloodline_switch_to_endless_tab")
class BloodlineSwitchToEndlessTabAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        try:
            params = json.loads(argv.custom_action_param or "{}")
        except json.JSONDecodeError:
            logger.info("切到无尽深渊: custom_action_param 非法，使用默认参数")
            params = {}
        if not isinstance(params, dict):
            params = {}

        max_wait_sec = _parse_float(params, "max_wait_sec", 20.0)
        loop_sleep_sec = _parse_float(params, "loop_sleep_sec", 0.2)
        post_click_delay_sec = _parse_float(params, "post_click_delay_sec", 2.0)
        started_at = time.monotonic()
        failed_clicked_count = 0

        while True:
            elapsed_sec = time.monotonic() - started_at
            if max_wait_sec > 0 and elapsed_sec >= max_wait_sec:
                logger.info(f"切到无尽深渊: 等待无尽深渊 tab 超时 {elapsed_sec:.1f}s")
                return False

            image = _safe_screencap(context)
            if image is None:
                if loop_sleep_sec > 0:
                    time.sleep(loop_sleep_sec)
                continue

            if _battle_failed_visible(context, image):
                failed_clicked_count += 1
                if _click_battle_failed_confirm(
                    context,
                    image,
                    attempt_no=failed_clicked_count,
                    threshold=0.8,
                    settle_sec=0.8,
                ):
                    logger.info(
                        f"切到无尽深渊: 先关闭战斗失败弹窗第 {failed_clicked_count} 次"
                    )
                    continue

            if _battle_end_visible(context, image):
                if _click_template_center(
                    context,
                    image,
                    BATTLE_END_CONFIRM_TEMPLATE,
                    settle_sec=0.5,
                ):
                    logger.info("切到无尽深渊: 先关闭战斗结束弹窗")
                    continue

            if _battle_selection_visible(context, image):
                logger.info("切到无尽深渊: 仍在战斗选择界面，交给战斗兜底处理")
                context.override_next(argv.node_name, ["血缘战斗兜底"])
                return True

            if _match_template(context, image, EXIT_CONFIRM_TEMPLATE):
                _click_rect_center(context, EXIT_CONFIRM_TARGET, "退出战斗确认")
                time.sleep(0.5)
                continue

            if _click_template_center(
                context,
                image,
                ENDLESS_TAB_TEMPLATE,
                settle_sec=post_click_delay_sec,
            ):
                logger.info("切到无尽深渊: 已点击无尽深渊 tab")
                return True

            if loop_sleep_sec > 0:
                time.sleep(loop_sleep_sec)


@AgentServer.custom_action("bloodline_switch_to_bloodline_tab")
class BloodlineSwitchToBloodlineTabAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        try:
            params = json.loads(argv.custom_action_param or "{}")
        except json.JSONDecodeError:
            logger.info("切回血缘: custom_action_param 非法，使用默认参数")
            params = {}
        if not isinstance(params, dict):
            params = {}

        max_wait_sec = _parse_float(params, "max_wait_sec", 10.0)
        loop_sleep_sec = _parse_float(params, "loop_sleep_sec", 0.2)
        post_click_delay_sec = _parse_float(params, "post_click_delay_sec", 1.0)
        started_at = time.monotonic()
        failed_clicked_count = 0

        while True:
            elapsed_sec = time.monotonic() - started_at
            if max_wait_sec > 0 and elapsed_sec >= max_wait_sec:
                logger.info(f"切回血缘: 等待血缘页稳定超时 {elapsed_sec:.1f}s")
                return False

            image = _safe_screencap(context)
            if image is None:
                if loop_sleep_sec > 0:
                    time.sleep(loop_sleep_sec)
                continue

            if _battle_failed_visible(context, image):
                failed_clicked_count += 1
                if _click_battle_failed_confirm(
                    context,
                    image,
                    attempt_no=failed_clicked_count,
                    threshold=0.8,
                    settle_sec=0.8,
                ):
                    logger.info(f"切回血缘: 先关闭战斗失败弹窗第 {failed_clicked_count} 次")
                    continue

            if _battle_end_visible(context, image):
                if _click_template_center(
                    context,
                    image,
                    BATTLE_END_CONFIRM_TEMPLATE,
                    settle_sec=0.5,
                ):
                    logger.info("切回血缘: 先关闭战斗结束弹窗")
                    continue

            if _match_template(context, image, EXIT_CONFIRM_TEMPLATE):
                _click_rect_center(context, EXIT_CONFIRM_TARGET, "退出战斗确认")
                time.sleep(0.5)
                continue

            if _battle_selection_visible(context, image):
                logger.info("切回血缘: 仍在战斗选择界面，交给战斗兜底处理")
                context.override_next(argv.node_name, ["血缘战斗兜底"])
                return True

            if _bloodline_stage_list_visible(context, image):
                logger.info("切回血缘: 已确认血缘关卡列表")
                return True

            if _click_template_center(
                context,
                image,
                BLOODLINE_TAB_TEMPLATE,
                threshold=0.8,
                settle_sec=post_click_delay_sec,
            ):
                logger.info("切回血缘: 已点击血缘 tab，继续确认页面")
                continue

            if loop_sleep_sec > 0:
                time.sleep(loop_sleep_sec)
