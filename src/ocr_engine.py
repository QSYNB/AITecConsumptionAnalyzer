import pytesseract
from PIL import Image
import re
import os

#1. 配置 Tesseract 路径 (本地运行可能需要)
#如果是 Windows, 请取消下行注释并指向你的 .exe 路径
pytesseract.pytesseract.tesseract_cmd = r"D:\15_MAI\7002\ocr\tesseract.exe"

def ocr_image(image_file):
    """
    运行 OCR 识别图片文字
    :param image_file: Streamlit 上传的 file_uploader 对象或图片路径
    :return: (full_text, lines)
    """
    try:
        # 打开图片
        img = Image.open(image_file)
        
        # 核心 OCR 动作
        raw_text = pytesseract.image_to_string(img)
        
        # 基础后处理：切分行并去除空格
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        clean_text = "\n".join(lines)
        
        return clean_text, lines
    except Exception as e:
        print(f"OCR Error: {e}")
        return "", []

def extract_total(lines):
    """
    从 OCR 提取的行中寻找总金额 (Total Amount)
    """
    # 关键词库
    TOTAL_KEYWORDS = [
        "total", "grand total", "net total", 
        "amount due", "amount", "balance", "total amount"
    ]
    
    # 金额正则表达式：匹配两位小数的数字
    price_pattern = r"\d+\.\d{2}"

    # 策略 1: 关键词搜索
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(k in line_lower for k in TOTAL_KEYWORDS):
            # 优先看同行是否有数字
            nums = re.findall(price_pattern, line)
            if nums:
                return nums[-1] # 通常总额是该行最后一个数
            
            # 如果同行没数字，看下一行
            if i + 1 < len(lines):
                nums_next = re.findall(price_pattern, lines[i+1])
                if nums_next:
                    return nums_next[0]

    # 策略 2: 兜底逻辑 - 提取整张收据的最后一个货币格式数字
    all_numbers = []
    for line in lines:
        nums = re.findall(price_pattern, line)
        all_numbers.extend(nums)
    
    if all_numbers:
        return all_numbers[-1]
    
    return "Not Found"

def extract_candidate_items(lines):
    candidates = []
    # 1. 扩充黑名单：增加收据中常见的非商品关键词
    blacklist = [
        "total", "subtotal", "tax", "gst", "sst", "cash", "change", 
        "address", "tel", "phone", "invoice", "receipt", "date", 
        "time", "table", "cashier", "server", "pax", "sdn bhd", "jalan"
    ]
    
    # 2. 增加正则匹配模式：用于识别并剔除日期、电话、发票号
    # 匹配日期 (如 05 Mar 2018)
    date_pattern = r"\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{4}"
    # 匹配编号或电话 (如 267388-U 或长数字)
    noise_pattern = r"(?:\d+-\w+)|(?:\d{10,})"

    for line in lines:
        line_low = line.lower()
        
        # 逻辑过滤：
        # - 不在黑名单内
        # - 不符合日期或纯编号模式
        # - 必须包含至少一个字母（防止纯符号行）
        if not any(k in line_low for k in blacklist):
            if not re.search(date_pattern, line_low) and not re.search(noise_pattern, line_low):
                if len(line) > 3 and any(c.isalpha() for c in line):
                    # 移除价格部分，仅保留商品描述
                    clean_line = re.sub(r"\s+\d{1,3}(?:,\d{3})*(?:\.\d{2})\s*$", "", line)
                    candidates.append(clean_line.strip())
                    
    return candidates