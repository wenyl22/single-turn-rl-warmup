import threading
import time
import sys
from typing import Tuple, Any, Optional
import traceback
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError


def execute_code(python_code_str: str, json_state: dict, timeout: int = 60) -> Tuple[bool, Any]:
    """
    执行 Python 代码字符串，并在指定时间内返回结果。
    
    Args:
        python_code_str (str): 包含 Python 函数定义的代码字符串
        json_state (dict): 传递给函数的 JSON 状态对象
        timeout (int): 超时时间（秒），默认为 60 秒
    
    Returns:
        Tuple[bool, Any]: (是否成功执行, 执行结果或错误信息)
        - 如果成功: (True, 函数返回值)
        - 如果失败: (False, 错误信息字符串)
    """
    
    def _execute_function(code_str, state):
        """实际执行函数的内部方法"""
        # 创建一个安全的执行环境
        # 只允许必要的内置函数，避免安全风险
        safe_globals = {}
        
        # 执行代码字符串，将函数定义加载到命名空间
        exec(code_str, safe_globals)
        
        # 检查是否定义了 next_action 函数
        if 'next_action' not in safe_globals:
            raise ValueError("代码中未找到 'next_action' 函数定义")
        
        # 获取函数引用
        next_action_func = safe_globals['next_action']
        
        # 验证输入参数
        if not isinstance(state, dict):
            raise TypeError("json_state 参数必须是字典类型")
        
        # 执行函数
        return next_action_func(state)
    
    try:
        # 使用 ThreadPoolExecutor 实现超时控制
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_execute_function, python_code_str, json_state)
            
            try:
                # 等待结果，设置超时
                result = future.result(timeout=timeout)
                return True, result
                
            except TimeoutError:
                return False, f"代码执行超时（超过 {timeout} 秒）"
        
            except Exception as e:
                # 处理在线程中发生的异常
                return False, f"执行异常: {type(e).__name__}: {str(e)}"
    except SyntaxError as e:
        return False, f"语法错误: {str(e)}"
    
    except Exception as e:
        # 捕获所有其他异常
        error_msg = f"运行时错误: {type(e).__name__}: {str(e)}"
        return False, error_msg
   