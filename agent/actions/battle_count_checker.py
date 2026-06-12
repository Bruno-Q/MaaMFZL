import json  
from typing import Optional
from maa.agent.agent_server import AgentServer  
from maa.custom_action import CustomAction  
from maa.context import Context  
from utils import logger
  
@AgentServer.custom_action("battle_count_checker")  
class BattleCountChecker(CustomAction):  
    _count: int = 0  # 当前已完成次数（实例级，整个任务共享）  
    _last_task_id: Optional[str] = None
  
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        task_id = argv.task_detail.task_id  
        if task_id != self._last_task_id: 
            self._count = 0          # 新任务，重置计数  
            self._last_task_id = task_id  
        
        # 从 pipeline 的 custom_action_param 读取目标次数  
        param = json.loads(argv.custom_action_param) if argv.custom_action_param else {}  
        target = int(param.get("target_count", 0))  
  
        self._count += 1  
        logger.info(f"[战斗结束] 已完成 {self._count} 次 / 目标 {'∞' if target == 0 else target} 次")  
  
        if target > 0 and self._count >= target:  
            logger.info(f"[战斗结束] 达到目标 {target} 次，停止任务")  
            context.override_next(argv.node_name, [])  # 清空 next，任务自然结束  
  
        return True  
