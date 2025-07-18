def dict_to_string_custom(data, indent=4, current_indent=0):
    if isinstance(data, dict):
        if not data:
            return "{}"
        
        result = "{\n"
        items = list(data.items())
        for i, (key, value) in enumerate(items):
            spaces = " " * (current_indent + indent)
            if isinstance(key, str):
                key_str = f'"{key}"'
            else:
                key_str = str(key)
            
            value_str = dict_to_string_custom(value, indent, current_indent + indent)
            result += f"{spaces}{key_str}: {value_str}"
            
            if i < len(items) - 1:
                result += ","
            result += "\n"
        
        result += " " * current_indent + "}"
        return result
    
    elif isinstance(data, list):
        if not data:
            return "[]"
        
        result = "["
        for i, item in enumerate(data):
            if i > 0:
                result += ", "
            result += dict_to_string_custom(item, indent, current_indent)
        result += "]"
        return result
    
    elif isinstance(data, str):
        return f'"{data}"'
    
    elif data is None:
        return "None"
    
    else:
        return str(data)