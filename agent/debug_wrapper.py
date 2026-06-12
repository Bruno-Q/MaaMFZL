#!/usr/bin/env python3  
import sys  
import os  
import debugpy


def _env_flag(name, default=False):
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _ensure_project_root_on_path():
    # 当前文件在 agent/ 下，父目录就是项目根目录
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root

  
def main():  
    print("=== debug_wrapper.py starting ===")  
    print(f"Python executable: {sys.executable}")  
    print(f"Working directory: {os.getcwd()}")  
    print(f"Arguments: {sys.argv}")  
    project_root = _ensure_project_root_on_path()
    print(f"Project root added to sys.path: {project_root}")
      
    debug_host = os.getenv("DEBUGPY_HOST", "127.0.0.1")
    debug_port = int(os.getenv("DEBUGPY_PORT", "5678"))
    wait_for_client = _env_flag("DEBUGPY_WAIT_FOR_CLIENT", default=True)
    continue_on_error = _env_flag("DEBUGPY_CONTINUE_ON_ERROR", default=False)

    print(f"Starting debugger on {debug_host}:{debug_port}...")
    try:
        debugpy.listen((debug_host, debug_port))
        print(f"Debugger listening on {debug_host}:{debug_port}")
    except Exception as e:
        print(f"Debugger error: {e}")
        if continue_on_error:
            print("DEBUGPY_CONTINUE_ON_ERROR=1, continuing without debugger...")
        else:
            print("Debugger is required. Exiting now. (Set DEBUGPY_CONTINUE_ON_ERROR=1 to bypass)")
            sys.exit(2)

    if wait_for_client:
        print("Waiting for debugger to connect...")
        debugpy.wait_for_client()
        print("Debugger connected! Continuing execution...")
    else:
        print("DEBUGPY_WAIT_FOR_CLIENT=0, not waiting for debugger connection.")
      
    # 导入并运行实际的 AgentServer  
    print("Importing agent.main...")  
    try:
        from agent.main import main
    except ModuleNotFoundError:
        # 兜底：当以脚本目录执行时，直接导入同目录 main
        from main import main
    print("Calling agent.main.main()...")  
    main()  
  
if __name__ == "__main__":  
    main()