from markitdown import MarkItDown

def convert_with_markitdown(input_pdf, output_md):
    md = MarkItDown()
    # 執行轉換
    result = md.convert(input_pdf)
    
    # 取得轉換後的文字內容
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(result.text_content)
    print(f"Markitdown 轉換完成：{output_md}")

# 執行
convert_with_markitdown("example.pdf", "output_markitdown.md")