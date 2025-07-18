import json
import signal
import threading
from typing import Dict, Any, Union, Tuple

def execute_code(python_code_str: str, json_state: Union[str, Dict[str, Any]], timeout: int = 60) -> Tuple[bool, Any]:
    """
    执行包含next_action函数的Python代码字符串，支持超时控制
    
    Args:
        python_code_str: 包含next_action函数的Python代码字符串
        json_state: JSON状态对象，可以是字符串或字典
        timeout: 超时时间（秒），默认60秒
    
    Returns:
        (success, result): 
        - success: bool, 是否执行成功
        - result: 执行结果(成功时)或错误信息(失败时)
    """
    
    # 存储执行结果的容器
    result_container = {'success': False, 'result': None, 'finished': False}
    
    def target_function():
        try:
            # 处理json_state
            if isinstance(json_state, str):
                parsed_state = json.loads(json_state)
            else:
                parsed_state = json_state
            
            # 创建安全的执行环境
            safe_globals = {'__builtins__': {}}
            
            # 执行代码并获取next_action函数
            exec(python_code_str, safe_globals)
            
            if 'next_action' not in safe_globals:
                result_container['success'] = False
                result_container['result'] = "代码中未找到next_action函数"
                return
            
            # 执行next_action函数
            result = safe_globals['next_action'](parsed_state)
            result_container['success'] = True
            result_container['result'] = result
            
        except Exception as e:
            result_container['success'] = False
            result_container['result'] = f"{type(e).__name__}: {str(e)}"
        finally:
            result_container['finished'] = True
    
    # 创建并启动线程
    thread = threading.Thread(target=target_function)
    thread.daemon = True  # 守护线程，主程序退出时自动结束
    thread.start()
    
    # 等待线程完成或超时
    thread.join(timeout)
    
    if thread.is_alive():
        # 超时了，线程仍在运行
        return False, f"代码执行超时（超过{timeout}秒）"
    elif result_container['finished']:
        # 正常完成
        return result_container['success'], result_container['result']
    else:
        # 异常情况
        return False, "代码执行异常终止"
    