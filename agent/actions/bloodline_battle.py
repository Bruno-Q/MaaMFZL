from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json  
from utils import logger

# 与 run_yolo_detect 中 pipeline threshold 保持一致
YOLO_CONF_THRESHOLD = 0.7
# {0: '金', 1: '远古', 2: '紫', 3: '太古'}
TAIGU_LABEL_INDEX = 3
TAIGU_CONFIRM_SLEEP_SEC = 0.05
TAIGU_CONFIRM_MAX_ATTEMPTS = 6
TAIGU_CONFIRM_INTERVAL_SEC = 0.03
BLOODLINE_BATTLE_MAX_DURATION_SEC = 240.0
FINAL_BOSS_THRESHOLD = 0.5
FINAL_BOSS_CONFIRM_REQUIRED_HITS = 2
FINAL_BOSS_CONFIRM_MAX_ATTEMPTS = 3
FINAL_BOSS_CONFIRM_INTERVAL_SEC = 0.05
MAX_CONSECUTIVE_SCREENCAP_FAILURES = 3
SCREENCAP_RETRY_INTERVAL_SEC = 0.15
CONTROL_IMMUNE_AFFIX_ROI = [1046, 634, 220, 35]
CONTROL_IMMUNE_AFFIX_KEYWORDS = ["免控"]
CONTROL_IMMUNE_AFFIX_OCR_THRESHOLD = 0.3
NON_CONTROL_IMMUNE_YOLO_ROI = [245, 52, 734, 661]


def _yolo_filtered_results(
    reco_detail,
    min_score: float = YOLO_CONF_THRESHOLD,
    roi: Optional[list[int]] = None,
) -> list:
    """返回达到置信度阈值的检测框。

    MAA 的 all_results 对应引擎 detail 里的 all（含低分框）；
    threshold 只作用于 filtered_results。业务逻辑必须用后者。
    """
    if not reco_detail:
        return []
    filtered = getattr(reco_detail, "filtered_results", None) or []
    if filtered:
        return [
            r
            for r in filtered
            if _box_center_in_roi(getattr(r, "box", [0, 0, 0, 0]), roi)
        ]
    return [
        r
        for r in (reco_detail.all_results or [])
        if getattr(r, "score", 0.0) >= min_score
        and _box_center_in_roi(getattr(r, "box", [0, 0, 0, 0]), roi)
    ]


def _load_chinese_font(size: int = 16):
    """加载支持中文的系统字体；失败则退回默认字体（可能仍无法显示中文）。"""
    candidates = [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        p = Path(path)
        if not p.is_file():
            continue
        try:
            return ImageFont.truetype(str(p), size)
        except OSError:
            continue
    return ImageFont.load_default()


def _numpy_to_pil_rgb(image: np.ndarray) -> Image.Image:
    draw_image = image.copy()
    if draw_image.ndim == 3 and draw_image.shape[2] == 3:
        # MAA 常见帧格式是 BGR，这里转为 RGB 供 PIL 保存/绘制。
        draw_image = draw_image[:, :, ::-1]
    return Image.fromarray(draw_image)


def _yolo_debug_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def build_yolo_debug_paths(
    node_name: str,
    timestamp: Optional[str] = None,
) -> tuple[Path, Path]:
    ts = timestamp or _yolo_debug_timestamp()
    safe_name = node_name or "yolo"
    stem = f"yolo_{safe_name}_{ts}"
    debug_dir = Path("debug")
    return debug_dir / f"{stem}_orig.jpg", debug_dir / f"{stem}_boxed.jpg"


def save_original_image(image: np.ndarray, output_path: Union[str, Path]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _numpy_to_pil_rgb(image).save(str(output_path))
    logger.info(f"Saved original image: {output_path}")


def save_yolo_debug_images(
    image: np.ndarray,
    results: list,
    node_name: str,
    roi: Optional[list[int]] = None,
) -> None:
    """识别到参数指定的期望类别时，保存原版与标注图各一份（文件名含时间戳）。"""
    if not results:
        return
    ts = _yolo_debug_timestamp()
    orig_path, boxed_path = build_yolo_debug_paths(node_name, ts)
    save_original_image(image, orig_path)
    draw_yolo_results(image, results, boxed_path, roi=roi)


def draw_yolo_results(
    image: np.ndarray,
    results: list,
    output_path: Union[str, Path],
    roi: Optional[list[int]] = None,
) -> None:
    """使用 PIL 绘制检测结果并保存到文件。"""
    if not results:
        return

    pil_img = _numpy_to_pil_rgb(image)
    draw = ImageDraw.Draw(pil_img)
    font = _load_chinese_font(16)

    if roi is not None:
        rx, ry, rw, rh = roi
        draw.rectangle(
            [(rx, ry), (rx + rw, ry + rh)],
            outline=(255, 220, 0),
            width=2,
        )

    for result in results:
        box = result.box
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h

        # 绘制检测框（红色）
        draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=2)

        # 添加标签
        label = f"{result.label}: {result.score:.2f}"
        text_height = 20
        ty = max(0, y1 - text_height - 4)
        draw.text((x1, ty), label, font=font, fill=(255, 0, 0))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(str(output_path))
    logger.info(f"Saved debug image: {output_path}")


def _box_center_in_roi(box, roi: Optional[list[int]]) -> bool:
    """用本地坐标过滤 YOLO 结果，避免依赖 NeuralNetworkDetect 内建 ROI 行为。"""
    if roi is None:
        return True
    if not box:
        return False
    x, y, w, h = box
    rx, ry, rw, rh = roi
    center_x = x + w / 2
    center_y = y + h / 2
    return rx <= center_x <= rx + rw and ry <= center_y <= ry + rh


def _log_detected_labels(
    reco_detail,
    min_score: float = YOLO_CONF_THRESHOLD,
    roi: Optional[list[int]] = None,
) -> None:
    """打印本次 YOLO 在阈值以上的全部检测（不限于业务期望类别）。"""
    results = _yolo_filtered_results(reco_detail, min_score, roi=roi)
    if not results:
        logger.info("识别到标签: 无")
        return

    parts = [
        f"{result.label}({result.cls_index}):{result.score:.2f}@{result.box}"
        for result in results
    ]
    if roi is not None:
        logger.info(f"ROI 内识别到标签 (共 {len(parts)} 个): {', '.join(parts)}")
    else:
        logger.info(f"识别到标签 (共 {len(parts)} 个): {', '.join(parts)}")


def _normalize_expected_label_indices(expected_labels) -> list[int]:
    """将参数中的期望类别统一为 int 列表。"""
    if expected_labels is None:
        return [1]
    if isinstance(expected_labels, int):
        return [expected_labels]
    if isinstance(expected_labels, list):
        out: list[int] = []
        for x in expected_labels:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                continue
        return out if out else [1]
    try:
        return [int(expected_labels)]
    except (TypeError, ValueError):
        return [1]


def _yolo_expected_results(
    reco_detail,
    expected_labels,
    min_score: float = YOLO_CONF_THRESHOLD,
    roi: Optional[list[int]] = None,
) -> list:
    """返回达到置信度阈值且属于期望类别的检测框。"""
    want = set(_normalize_expected_label_indices(expected_labels))
    return [
        r
        for r in _yolo_filtered_results(reco_detail, min_score, roi=roi)
        if getattr(r, "cls_index", -1) in want
    ]


def _yolo_hit_expected(
    reco_detail,
    expected_labels,
    min_score: float = YOLO_CONF_THRESHOLD,
    roi: Optional[list[int]] = None,
) -> bool:
    """在达到置信度阈值的检测结果中判断是否包含任一期望类别。"""
    return bool(_yolo_expected_results(reco_detail, expected_labels, min_score, roi=roi))


def _requires_taigu_double_confirm(expected_labels) -> bool:
    """仅检测太古时，需连续两帧均命中（掉落初帧其他品质会短暂同色）。"""
    return _normalize_expected_label_indices(expected_labels) == [TAIGU_LABEL_INDEX]


def _confirm_taigu_hit_burst(
    context: Context,
    reco_detail,
    image: np.ndarray,
    *,
    debug_node_name: str = "yolo",
    save_debug: bool = False,
    confirm_sleep_sec: float = TAIGU_CONFIRM_SLEEP_SEC,
    max_attempts: int = TAIGU_CONFIRM_MAX_ATTEMPTS,
    interval_sec: float = TAIGU_CONFIRM_INTERVAL_SEC,
    roi: Optional[list[int]] = None,
) -> bool:
    """太古首次命中后，短等 + 快速连拍补帧，任一帧再命中即确认成功。"""
    expected = [TAIGU_LABEL_INDEX]
    if not _yolo_hit_expected(reco_detail, expected, roi=roi):
        return False

    logger.info("太古首次命中，开始快速二次确认...")
    if save_debug:
        save_yolo_debug_images(
            image,
            _yolo_expected_results(reco_detail, expected, roi=roi),
            debug_node_name,
            roi=roi,
        )

    if confirm_sleep_sec > 0:
        time.sleep(confirm_sleep_sec)

    for attempt in range(1, max_attempts + 1):
        confirm_image = safe_screencap(context)
        if confirm_image is None:
            logger.info(f"太古二次确认第 {attempt} 次截图失败")
            continue

        confirm_reco = run_yolo_detect(
            context,
            confirm_image,
            verbose=False,
            debug_node_name=debug_node_name,
            expected_labels=expected,
            roi=roi,
        )
        if _yolo_hit_expected(confirm_reco, expected, roi=roi):
            logger.info(f"太古连续确认命中（补帧第 {attempt} 次）")
            if save_debug:
                save_yolo_debug_images(
                    confirm_image,
                    _yolo_expected_results(confirm_reco, expected, roi=roi),
                    debug_node_name,
                    roi=roi,
                )
            return True

        if attempt < max_attempts and interval_sec > 0:
            time.sleep(interval_sec)

    logger.info("太古二次确认未命中，视为误检（掉落初帧颜色干扰）")
    return False


def _confirm_yolo_hit(
    context: Context,
    reco_detail,
    image: np.ndarray,
    expected_labels,
    *,
    debug_node_name: str = "yolo",
    save_debug: bool = False,
    confirm_sleep_sec: float = TAIGU_CONFIRM_SLEEP_SEC,
    confirm_max_attempts: int = TAIGU_CONFIRM_MAX_ATTEMPTS,
    confirm_interval_sec: float = TAIGU_CONFIRM_INTERVAL_SEC,
    roi: Optional[list[int]] = None,
    taigu_double_confirm: bool = True,
) -> bool:
    """确认 YOLO 命中；期望类别仅为太古时可启用连续两帧确认。"""
    if not _yolo_hit_expected(reco_detail, expected_labels, roi=roi):
        return False

    if not taigu_double_confirm or not _requires_taigu_double_confirm(expected_labels):
        if save_debug:
            save_yolo_debug_images(
                image,
                _yolo_expected_results(reco_detail, expected_labels, roi=roi),
                debug_node_name,
                roi=roi,
            )
        return True

    return _confirm_taigu_hit_burst(
        context,
        reco_detail,
        image,
        debug_node_name=debug_node_name,
        save_debug=save_debug,
        confirm_sleep_sec=confirm_sleep_sec,
        max_attempts=confirm_max_attempts,
        interval_sec=confirm_interval_sec,
        roi=roi,
    )


def run_yolo_detect(
    context: Context,
    image: np.ndarray,
    draw_result: bool = False,
    debug_node_name: str = "yolo",
    expected_labels=None,
    verbose: bool = True,
    roi: Optional[list[int]] = None,
):
    """对单帧图像执行一次 YOLO 检测，并按需绘制结果图。

    不在 pipeline 中传入 expected，以便 filtered_results 包含各期望类别；
    是否命中「期望掉落」请用 _yolo_hit_expected(..., roi=roi)。
    注意勿用 all_results，其中含未过 threshold 的低分框。
    """

    t0 = time.perf_counter()
    yolo_detect_config = {
        "recognition": "NeuralNetworkDetect",
        "model": "yolov8n.onnx",
        # {0: '金', 1: '远古', 2: '紫'， 3: '太古'} — 不传 expected，保留全部分类供日志与调试图
        "threshold": YOLO_CONF_THRESHOLD,
    }
    reco_detail = context.run_recognition(
        "YoloDetect",
        image,
        pipeline_override={
            "YoloDetect": yolo_detect_config,
        },
    )
    yolo_ms = (time.perf_counter() - t0) * 1000
    filtered = _yolo_filtered_results(reco_detail, roi=roi)
    if verbose or filtered:
        logger.info(f"YoloDetect 耗时: {yolo_ms:.2f} ms")
        if roi is not None:
            logger.info(f"YOLO 使用本地 ROI 过滤: {roi}")
        _log_detected_labels(reco_detail, roi=roi)

    if (
        draw_result
        and expected_labels is not None
        and not _requires_taigu_double_confirm(expected_labels)
    ):
        save_yolo_debug_images(
            image,
            _yolo_expected_results(reco_detail, expected_labels, roi=roi),
            debug_node_name,
            roi=roi,
        )

    return reco_detail

# 检测是否出现最终boss
def detect_final_boss(
    context: Context,
    image: np.ndarray,
    *,
    threshold: float = FINAL_BOSS_THRESHOLD,
    verbose: bool = True,
):
    reco_detail = context.run_recognition(
        "FeatureDetect",
        image,
        pipeline_override={
            "FeatureDetect": {
                "recognition": "TemplateMatch",
                "template": "出现最终boss.png",
                # 这里调低一些阈值，因为最终boss出现的时候，可能会有一些其他的干扰
                "threshold": threshold,
            }
        },
    )
    if reco_detail and reco_detail.hit:
        if verbose:
            logger.info("出现最终boss")
        return True
    else:
        if verbose:
            logger.info("未出现最终boss")
        return False

# 检测战斗失败
def detect_battle_failed(context: Context, image: np.ndarray):
    reco_detail = context.run_recognition(  
        "FeatureDetect",  
        image,  
        pipeline_override={  
            "FeatureDetect": {  
                "recognition": "TemplateMatch",  
                "template": "战斗失败.png",  
                "threshold": 0.8  
            }  
        }  
    )  
    if reco_detail and reco_detail.hit:
        logger.info("战斗失败")
        return True
    else:
        logger.info("未战斗失败")
        return False


def detect_control_immune_affix(
    context: Context,
    image: np.ndarray,
) -> bool:
    """OCR 打印小怪词条，并检测是否带有免控词条。"""
    logger.info(f"小怪词条 OCR ROI: {CONTROL_IMMUNE_AFFIX_ROI}")
    reco_detail = context.run_recognition(
        "ControlImmuneAffixOCR",
        image,
        pipeline_override={
            "ControlImmuneAffixOCR": {
                "recognition": "OCR",
                "roi": CONTROL_IMMUNE_AFFIX_ROI,
                "threshold": CONTROL_IMMUNE_AFFIX_OCR_THRESHOLD,
            }
        },
    )

    recognized_texts = []
    if reco_detail and reco_detail.all_results:
        recognized_texts = [
            str(result.text).strip()
            for result in reco_detail.all_results
            if getattr(result, "text", None)
        ]

    logger.info(f"检测到的小怪词条: {recognized_texts or '无'}")
    has_control_immune_affix = any(
        keyword in text
        for keyword in CONTROL_IMMUNE_AFFIX_KEYWORDS
        for text in recognized_texts
    )
    if has_control_immune_affix:
        logger.info(f"检测到小怪免控词条: {CONTROL_IMMUNE_AFFIX_KEYWORDS}")
        return True

    logger.info("未检测到小怪免控词条")
    return False


def _manual_click_by_template(
    context: Context,
    image: np.ndarray,
    template: str,
    threshold: float = 0.8,
    post_click_settle_sec: float = 1.0,
) -> bool:
    """模板匹配命中后，手动点击识别框中心点。"""
    reco_detail = context.run_recognition(
        "TemplateMatch",
        image,
        pipeline_override={
            "TemplateMatch": {
                "recognition": "TemplateMatch",
                "template": template,
                "threshold": threshold,
            }
        },
    )
    if not (reco_detail and reco_detail.hit and reco_detail.box):
        logger.info(f"未匹配到可点击目标: {template}")
        return False

    x, y, w, h = reco_detail.box
    center_x = x + w // 2
    center_y = y + h // 2
    logger.info(f"点击 {template} 位置: ({center_x}, {center_y})")
    context.tasker.controller.post_click(center_x, center_y).wait()
    # 点击后给设备/画面一点稳定时间，避免紧接着截图卡住
    if post_click_settle_sec > 0:
        time.sleep(post_click_settle_sec)
    return True


def safe_screencap(context: Context) -> Optional[np.ndarray]:
    """截图并捕获异常，失败时返回 None。"""
    image = context.tasker.controller.post_screencap().wait().get()
    if image is None or not isinstance(image, np.ndarray):
        logger.info("截图失败: 返回空或类型异常")
        return None
    return image


def click_pause(context: Context, image: np.ndarray) -> bool:
    return _manual_click_by_template(
        context,
        image,
        template="暂停.png",
        threshold=0.8,
    )


def click_enter_final_boss(
    context: Context,
    image: np.ndarray,
) -> bool:
    return _manual_click_by_template(
        context,
        image,
        template="最终boss.png",
        threshold=0.8,
    )


def click_enable_one_speed(
    context: Context,
    image: np.ndarray,
) -> bool:
    # 页面显示“二倍速”按钮时，点击可切换到一倍速
    return _manual_click_by_template(
        context,
        image,
        template="二倍速.png",
        threshold=0.8,
    )


def _confirm_final_boss_hit(
    context: Context,
    image: np.ndarray,
    *,
    threshold: float = FINAL_BOSS_THRESHOLD,
    required_hits: int = FINAL_BOSS_CONFIRM_REQUIRED_HITS,
    max_attempts: int = FINAL_BOSS_CONFIRM_MAX_ATTEMPTS,
    interval_sec: float = FINAL_BOSS_CONFIRM_INTERVAL_SEC,
) -> bool:
    """最终 boss 出现提示需要短时二次确认，减少误判导致的异常退出。"""
    if required_hits <= 1:
        return detect_final_boss(context, image, threshold=threshold, verbose=True)

    if not detect_final_boss(context, image, threshold=threshold, verbose=True):
        return False

    hits = 1
    for attempt in range(2, max_attempts + 1):
        if interval_sec > 0:
            time.sleep(interval_sec)
        confirm_image = safe_screencap(context)
        if confirm_image is None:
            logger.info(f"最终boss二次确认第 {attempt} 次截图失败")
            continue
        if detect_final_boss(context, confirm_image, threshold=threshold, verbose=False):
            hits += 1
            if hits >= required_hits:
                logger.info(f"最终boss确认命中（{hits}/{required_hits}）")
                return True

    logger.info(f"最终boss提示仅命中 {hits}/{required_hits} 次，视为误判")
    return False

@AgentServer.custom_action("bloodline_battle")
class BloodlineBattleAction(CustomAction):

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        # 读取option参数
        params = json.loads(argv.custom_action_param or "{}") 
        yolo_expected_labels = _normalize_expected_label_indices(
            params.get("yolo_expected_labels", [1])
        )
        taigu_only = _requires_taigu_double_confirm(yolo_expected_labels)

        battle_loop_sleep_sec_raw = params.get(
            "battle_loop_sleep_sec",
            0.0 if taigu_only else 0.1,
        )
        try:
            battle_loop_sleep_sec = float(battle_loop_sleep_sec_raw)
        except (TypeError, ValueError):
            logger.info(
                f"battle_loop_sleep_sec 参数非法: {battle_loop_sleep_sec_raw!r}，"
                f"回退为默认值 {0.0 if taigu_only else 0.1}"
            )
            battle_loop_sleep_sec = 0.0 if taigu_only else 0.1

        def _float_param(key: str, default: float) -> float:
            raw = params.get(key, default)
            try:
                return float(raw)
            except (TypeError, ValueError):
                logger.info(f"{key} 参数非法: {raw!r}，回退为默认值 {default}")
                return default

        def _int_param(key: str, default: int) -> int:
            raw = params.get(key, default)
            try:
                return int(raw)
            except (TypeError, ValueError):
                logger.info(f"{key} 参数非法: {raw!r}，回退为默认值 {default}")
                return default

        taigu_confirm_sleep_sec = _float_param(
            "taigu_confirm_sleep_sec", TAIGU_CONFIRM_SLEEP_SEC
        )
        taigu_confirm_max_attempts = _int_param(
            "taigu_confirm_max_attempts", TAIGU_CONFIRM_MAX_ATTEMPTS
        )
        taigu_confirm_interval_sec = _float_param(
            "taigu_confirm_interval_sec", TAIGU_CONFIRM_INTERVAL_SEC
        )
        battle_max_duration_sec = _float_param(
            "battle_max_duration_sec", BLOODLINE_BATTLE_MAX_DURATION_SEC
        )
        final_boss_threshold = _float_param(
            "final_boss_threshold", FINAL_BOSS_THRESHOLD
        )
        final_boss_confirm_required_hits = _int_param(
            "final_boss_confirm_required_hits", FINAL_BOSS_CONFIRM_REQUIRED_HITS
        )
        final_boss_confirm_max_attempts = _int_param(
            "final_boss_confirm_max_attempts", FINAL_BOSS_CONFIRM_MAX_ATTEMPTS
        )
        final_boss_confirm_interval_sec = _float_param(
            "final_boss_confirm_interval_sec", FINAL_BOSS_CONFIRM_INTERVAL_SEC
        )
        max_consecutive_screencap_failures = _int_param(
            "max_consecutive_screencap_failures", MAX_CONSECUTIVE_SCREENCAP_FAILURES
        )
        screencap_retry_interval_sec = _float_param(
            "screencap_retry_interval_sec", SCREENCAP_RETRY_INTERVAL_SEC
        )

        logger.info(f"yolo_expected_labels: {yolo_expected_labels}")
        if taigu_only:
            logger.info(
                "太古模式: "
                f"loop_sleep={battle_loop_sleep_sec}s, "
                f"confirm_sleep={taigu_confirm_sleep_sec}s, "
                f"confirm_attempts={taigu_confirm_max_attempts}, "
                f"confirm_interval={taigu_confirm_interval_sec}s"
            )
        logger.info(
            "血缘战斗保护参数: "
            f"battle_max_duration={battle_max_duration_sec}s, "
            f"final_boss_threshold={final_boss_threshold}, "
            f"final_boss_confirm={final_boss_confirm_required_hits}/{final_boss_confirm_max_attempts}, "
            f"screencap_retries={max_consecutive_screencap_failures}"
        )

        # 点击进入最终boss关卡
        image = safe_screencap(context)
        if image is None:
            logger.info("首次截图失败，退出当前自定义动作")
            return False
        if not click_enter_final_boss(context, image):
            logger.info("未识别到最终boss入口，退出当前自定义动作")
            return False

        # 开启一倍速
        speed_image = safe_screencap(context)
        if speed_image is not None:
            click_enable_one_speed(context, speed_image)
        else:
            logger.info("切速前截图失败，跳过切速点击")

        yolo_success_count = 0
        loop_count = 0
        consecutive_screencap_failures = 0
        battle_started_at = time.monotonic()
        last_image = image

        # OCR检测小怪是否有免控词条
        affix_image = safe_screencap(context)
        has_control_immune_affix = False
        affix_detected = False
        if affix_image is not None:
            has_control_immune_affix = detect_control_immune_affix(
                context,
                affix_image,
            )
            affix_detected = True
        else:
            logger.info("免控词条检测截图失败，保持当前 YOLO 全图检测逻辑")

        yolo_detect_roi = None
        if affix_detected and not has_control_immune_affix:
            yolo_detect_roi = NON_CONTROL_IMMUNE_YOLO_ROI
        taigu_double_confirm_enabled = yolo_detect_roi is None

        if yolo_detect_roi is None:
            logger.info("检测到免控词条或检测状态不确定，保持当前 YOLO 全图检测逻辑")
        else:
            logger.info(
                f"未检测到免控词条，YOLO 仅检测 ROI: {yolo_detect_roi}，"
                "太古命中不进行二次确认"
            )
            if affix_image is not None:
                logger.info(f"YOLO ROI: {yolo_detect_roi}")

        while True:
            loop_count += 1
            if taigu_only and loop_count % 200 == 0:
                logger.info(
                    f"太古检测循环中... 已运行 {loop_count} 轮，"
                    f"确认命中 {yolo_success_count} 次"
                )

            elapsed_sec = time.monotonic() - battle_started_at
            if battle_max_duration_sec > 0 and elapsed_sec >= battle_max_duration_sec:
                logger.info(
                    f"血缘战斗超时 {elapsed_sec:.1f}s，尝试暂停并退出当前战斗"
                )
                pause_img = safe_screencap(context)
                if pause_img is None:
                    pause_img = last_image
                click_pause(context, pause_img)
                return False

            new_image = safe_screencap(context)
            if new_image is None:
                consecutive_screencap_failures += 1
                logger.info(
                    f"循环截图失败，第 {consecutive_screencap_failures} 次"
                )
                if consecutive_screencap_failures >= max_consecutive_screencap_failures:
                    logger.info("循环截图连续失败次数过多，返回失败")
                    return False
                if screencap_retry_interval_sec > 0:
                    time.sleep(screencap_retry_interval_sec)
                continue
            consecutive_screencap_failures = 0
            last_image = new_image
            reco_detail = run_yolo_detect(
                context,
                new_image,
                draw_result=True,
                debug_node_name=argv.node_name,
                expected_labels=yolo_expected_labels,
                verbose=not taigu_only,
                roi=yolo_detect_roi,
            )
            if _confirm_yolo_hit(
                context,
                reco_detail,
                new_image,
                yolo_expected_labels,
                debug_node_name=argv.node_name,
                save_debug=True,
                confirm_sleep_sec=taigu_confirm_sleep_sec,
                confirm_max_attempts=taigu_confirm_max_attempts,
                confirm_interval_sec=taigu_confirm_interval_sec,
                roi=yolo_detect_roi,
                taigu_double_confirm=taigu_double_confirm_enabled,
            ):
                yolo_success_count += 1
                logger.info(f"YOLO 命中计数: {yolo_success_count}")

            if _confirm_final_boss_hit(
                context,
                new_image,
                threshold=final_boss_threshold,
                required_hits=final_boss_confirm_required_hits,
                max_attempts=final_boss_confirm_max_attempts,
                interval_sec=final_boss_confirm_interval_sec,
            ):
                if yolo_success_count >= 1:
                    return True
                if yolo_success_count == 0:
                    pause_img = safe_screencap(context)
                    if pause_img is None:
                        pause_img = new_image
                    click_pause(context, pause_img)
                    return False
            if detect_battle_failed(context, new_image):
                pause_img = safe_screencap(context)
                if pause_img is None:
                    pause_img = new_image
                click_pause(context, pause_img)
                return False

            time.sleep(battle_loop_sleep_sec)
