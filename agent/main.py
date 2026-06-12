import importlib.util
import sys
from pathlib import Path

from maa.agent.agent_server import AgentServer
from maa.tasker import Tasker
from utils import logger


def _auto_load_plugins() -> None:
    """自动加载 actions/recognitions 下的插件文件。"""
    base_dir = Path(__file__).resolve().parent
    plugin_dirs = ["actions", "recognitions"]
    loaded_modules = set()

    for folder in plugin_dirs:
        plugin_dir = base_dir / folder
        if not plugin_dir.is_dir():
            continue

        for file in sorted(plugin_dir.glob("*.py")):
            if file.name.startswith("_"):
                continue

            module_name = f"maa_{folder}_{file.stem}"
            if module_name in loaded_modules:
                continue

            spec = importlib.util.spec_from_file_location(module_name, file)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"无法加载插件: {file}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded_modules.add(module_name)

#Tasker.set_stdout_level(LoggingLevelEnum.Debug)  

def main():
    _auto_load_plugins()
    
    Tasker.set_log_dir("./debug")
    Tasker.set_save_on_error(False)

    if len(sys.argv) < 2:
        logger.error("Usage: python main.py <socket_id>")
        logger.error("socket_id is provided by AgentIdentifier.")
        sys.exit(1)
        
    socket_id = sys.argv[-1]

    logger.info(f"socket_id: {socket_id}")
    AgentServer.start_up(socket_id)
    AgentServer.join()
    AgentServer.shut_down()


if __name__ == "__main__":
    main()
