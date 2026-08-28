import json  
from typing import Optional
from maa.agent.agent_server import AgentServer  
from maa.custom_action import CustomAction  
from maa.context import Context  
from utils import logger

@AgentServer.custom_action("bloodline_count_checker")
class BloodlineCountChecker(CustomAction):
    _bloodline_count: int = 0  # 当前已完成次数（实例级，整个任务共享）  
    _bloodline_last_task_id: Optional[str] = None
  
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        task_id = argv.task_detail.task_id  
        if task_id != self._bloodline_last_task_id: 
            self._bloodline_count = 0          # 新任务，重置计数  
            self._bloodline_last_task_id = task_id  
        
        # 兼容历史拼写 battle_ltarget_count，优先使用标准字段 target_count。
        param = json.loads(argv.custom_action_param) if argv.custom_action_param else {}
        raw_target = param.get("target_count", param.get("battle_ltarget_count", 0))
        target = int(raw_target)
  
        self._bloodline_count += 1  
        logger.info(f"[战斗结束] 已完成 {self._bloodline_count} 次 / 目标 {'∞' if target == 0 else target} 次")  
  
        if target > 0 and self._bloodline_count >= target:  
            logger.info(f"[战斗结束] 达到目标 {target} 次，停止任务")  
            context.override_next(argv.node_name, [])  # 清空 next，任务自然结束  
  
        return True  
