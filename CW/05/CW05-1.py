import pdfplumber

def convert_with_pdfplumber(input_pdf, output_md):
    with pdfplumber.open(input_pdf) as pdf:
        full_text = ""
        for page in pdf.pages:
            # 提取每一頁的文字
            text = page.extract_text()
            if text:
                full_text += text + "\n\n--- PAGE BREAK ---\n\n"
        
        # 寫入 Markdown 檔案
        with open(output_md, "w", encoding="utf-8") as f:
            f.write(full_text)
    print(f"pdfplumber 轉換完成：{output_md}")

# 執行
convert_with_pdfplumber("example.pdf", "output_pdfplumber.md")