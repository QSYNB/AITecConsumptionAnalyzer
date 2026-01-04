import pytesseract
from PIL import Image
import re
import os


pytesseract.pytesseract.tesseract_cmd = r"D:\15_MAI\7002\ocr\tesseract.exe"

def ocr_image(image_file):
    """
    运行 OCR 识别图片文字
    :param image_file: Streamlit 上传的 file_uploader 对象或图片路径
    :return: (full_text, lines)
    """
    try:

        img = Image.open(image_file)
        

        raw_text = pytesseract.image_to_string(img)
        

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
    

    price_pattern = r"\d+\.\d{2}"


    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(k in line_lower for k in TOTAL_KEYWORDS):

            nums = re.findall(price_pattern, line)
            if nums:
                return nums[-1] 
            

            if i + 1 < len(lines):
                nums_next = re.findall(price_pattern, lines[i+1])
                if nums_next:
                    return nums_next[0]


    all_numbers = []
    for line in lines:
        nums = re.findall(price_pattern, line)
        all_numbers.extend(nums)
    
    if all_numbers:
        return all_numbers[-1]
    
    return "Not Found"

def extract_candidate_items(lines):
    candidates = []

    blacklist = [
        "total", "subtotal", "tax", "gst", "sst", "cash", "change", 
        "address", "tel", "phone", "invoice", "receipt", "date", 
        "time", "table", "cashier", "server", "pax", "sdn bhd", "jalan"
    ]
    

    date_pattern = r"\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{4}"

    noise_pattern = r"(?:\d+-\w+)|(?:\d{10,})"

    for line in lines:
        line_low = line.lower()
        

        if not any(k in line_low for k in blacklist):
            if not re.search(date_pattern, line_low) and not re.search(noise_pattern, line_low):
                if len(line) > 3 and any(c.isalpha() for c in line):
                 
                    clean_line = re.sub(r"\s+\d{1,3}(?:,\d{3})*(?:\.\d{2})\s*$", "", line)
                    candidates.append(clean_line.strip())
                    
    return candidates