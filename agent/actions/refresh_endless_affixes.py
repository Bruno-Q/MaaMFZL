from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
import time
from typing import Any

import numpy as np
import json
from utils import logger

@AgentServer.custom_action("refresh_endless_affixes")
class RefreshEndlessAffixesAction(CustomAction):
    DEFAULT_OCR_ROI = [775, 358, 276, 114]
    BATTLE_FAILED_CONFIRM_TEMPLATE = "战斗失败_确定.png"
    BATTLE_FAILED_CONFIRM_THRESHOLD = 0.8
    BATTLE_FAILED_RECOVERY_DELAY_SEC = 0.8

    @staticmethod
    def _safe_screencap(context: Context):
        image = context.tasker.controller.post_screencap().wait().get()
        if image is None or not isinstance(image, np.ndarray):
            return None
        return image

    @staticmethod
    def _extract_recognized_texts(
        context: Context,
        image: np.ndarray,
        roi: list[int],
    ) -> list[str]:
        reco_detail = context.run_recognition(
            "MyOCR",
            image,
            pipeline_override={
                "MyOCR": {
                    "recognition": "OCR",
                    "roi": roi,
                }
            },
        )

        recognized_texts: list[str] = []
        if reco_detail and reco_detail.hit and reco_detail.all_results:
            recognized_texts = [
                str(result.text).strip()
                for result in reco_detail.all_results
                if getattr(result, "text", None)
            ]
        return recognized_texts

    @staticmethod
    def _collect_matched_affixes(
        recognized_texts: list[str],
        blocked_affixes: list[str],
    ) -> list[str]:
        return [
            blocked
            for blocked in blocked_affixes
            if any(blocked in text for text in recognized_texts)
        ]

    @staticmethod
    def _click_refresh_by_template(
        context: Context,
        image: np.ndarray,
        template: str,
        threshold: float,
    ) -> bool:
        reco_detail = context.run_recognition(
            "RefreshTemplateMatch",
            image,
            pipeline_override={
                "RefreshTemplateMatch": {
                    "recognition": "TemplateMatch",
                    "template": template,
                    "threshold": threshold,
                }
            },
        )
        if not (reco_detail and reco_detail.hit and reco_detail.box):
            logger.info(f"未匹配到刷新按钮模板: {template}")
            return False

        x, y, w, h = reco_detail.box
        click_x = x + w // 2
        click_y = y + h // 2
        logger.info(f"点击刷新按钮: ({click_x}, {click_y})")
        context.tasker.controller.post_click(click_x, click_y).wait()
        return True

    @staticmethod
    def _click_battle_failed_confirm(
        context: Context,
        image: np.ndarray,
        template: str,
        threshold: float,
    ) -> bool:
        reco_detail = context.run_recognition(
            "BattleFailedConfirmTemplateMatch",
            image,
            pipeline_override={
                "BattleFailedConfirmTemplateMatch": {
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
        logger.info(f"刷新词条前检测到战斗失败弹窗，点击确定: ({click_x}, {click_y})")
        context.tasker.controller.post_click(click_x, click_y).wait()
        return True

    @staticmethod
    def _parse_config(custom_action_param: str) -> dict[str, Any]:
        if not custom_action_param:
            return {}
        try:
            parsed = json.loads(custom_action_param)
        except json.JSONDecodeError:
            logger.info("custom_action_param 不是合法 JSON，使用默认配置")
            return {}
        if isinstance(parsed, dict):
            return parsed
        logger.info("custom_action_param 解析结果不是对象，使用默认配置")
        return {}

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        params = self._parse_config(argv.custom_action_param or "")

        blocked_affixes_str = params.get("blocked_affixes", "")  
        blocked_affixes = [f.strip() for f in blocked_affixes_str.split(",") if f.strip()]  

        ocr_roi = self.DEFAULT_OCR_ROI
        refresh_template = "刷新.png"
        refresh_threshold = 0.8
        refresh_interval_sec = 0.4
        max_refresh_count = params.get("count", 10)
        max_empty_ocr_retry = 3

        image = self._safe_screencap(context)
        if image is None:
            if not blocked_affixes:
                logger.info("截图失败，但 blocked_affixes 为空，跳过刷新词条")
                return True
            logger.info("截图失败，停止刷新词条")
            return False

        if self._click_battle_failed_confirm(
            context=context,
            image=image,
            template=self.BATTLE_FAILED_CONFIRM_TEMPLATE,
            threshold=self.BATTLE_FAILED_CONFIRM_THRESHOLD,
        ):
            time.sleep(self.BATTLE_FAILED_RECOVERY_DELAY_SEC)

        if not blocked_affixes:
            logger.info("blocked_affixes 为空，无需刷新词条")
            return True

        refresh_count = 0
        empty_ocr_count = 0
        while True:
            image = self._safe_screencap(context)
            if image is None:
                logger.info("截图失败，停止刷新词条")
                return False

            if self._click_battle_failed_confirm(
                context=context,
                image=image,
                template=self.BATTLE_FAILED_CONFIRM_TEMPLATE,
                threshold=self.BATTLE_FAILED_CONFIRM_THRESHOLD,
            ):
                empty_ocr_count = 0
                time.sleep(self.BATTLE_FAILED_RECOVERY_DELAY_SEC)
                continue

            recognized_texts = self._extract_recognized_texts(context, image, ocr_roi)
            logger.info(f"当前识别词条: {recognized_texts}")
            if not recognized_texts:
                empty_ocr_count += 1
                if empty_ocr_count > max_empty_ocr_retry:
                    logger.info(f"OCR 连续空结果超过 {max_empty_ocr_retry} 次，停止刷新")
                    return False
                logger.info("OCR 未识别到文本，重试当前轮")
                time.sleep(refresh_interval_sec)
                continue
            empty_ocr_count = 0

            matched_affixes = self._collect_matched_affixes(recognized_texts, blocked_affixes)
            if not matched_affixes:
                logger.info("当前词条均符合要求，结束刷新")
                return True

            if not self._click_refresh_by_template(
                context=context,
                image=image,
                template=refresh_template,
                threshold=refresh_threshold,
            ):
                return False

            refresh_count += 1
            logger.info(f"第 {refresh_count} 次刷新，命中词条: {matched_affixes}")
            if refresh_interval_sec > 0:
                time.sleep(refresh_interval_sec)

        return True
