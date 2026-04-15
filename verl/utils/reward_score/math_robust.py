import re
from math_verify.metric import math_metric
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig

def extract_last_valid_boxed_content(text: str) -> str:
    """
    扫描文本，提取所有结构完整的 \boxed{...} 内容，并返回最后一个。
    支持嵌套括号，例如 \boxed{\frac{1}{2}}。
    """
    valid_contents = []
    
    # 查找所有 \boxed{ 的起始位置
    # 注意：这里我们手动解析，不依赖正则匹配内容，因为正则很难处理嵌套括号
    start_indices = [m.start() for m in re.finditer(r'\\boxed\{', text)]
    
    for start in start_indices:
        # 从 \boxed{ 的 { 之后开始计数
        content_start = start + 7 # len("\\boxed{")
        balance = 1
        current_content = []
        
        # 遍历后续字符寻找闭合的 }
        for i in range(content_start, len(text)):
            char = text[i]
            if char == '{':
                balance += 1
            elif char == '}':
                balance -= 1
            
            if balance == 0:
                # 找到匹配的右括号，记录内容并停止当前查找
                valid_contents.append("".join(current_content))
                break
            
            current_content.append(char)
            
    if valid_contents:
        return valid_contents[-1] # 返回最后一个有效的
    return None

def compute_score(model_output: str, ground_truth: str) -> float:
    # 1. 强制提取最后一个有效的 boxed 内容
    extracted_content = extract_last_valid_boxed_content(model_output)
    
    # 2. 如果提取到了，重新包装成标准的 \boxed{x} 格式
    #    如果没提取到（比如全是乱码），就保留原样让 math_verify 去处理（或者直接判0）
    if extracted_content is not None:
        cleaned_output = f"\\boxed{{{extracted_content}}}"
    else:
        return 0.

    # 3. 准备 Ground Truth
    ground_truth_boxed = f"\\boxed{{{ground_truth}}}" if "\\boxed" not in ground_truth else ground_truth

    # 4. 调用 math_verify
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    
    try:
        # 此时传入的 cleaned_output 已经是干净的 \boxed{271} 了
        ret_score, _ = verify_func([ground_truth_boxed], [cleaned_output])
    except Exception:
        ret_score = 0.
        
    return ret_score