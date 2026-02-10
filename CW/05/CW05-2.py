from docling.document_converter import DocumentConverter

def convert_with_docling(input_pdf, output_md):
    converter = DocumentConverter()
    # 執行轉換
    result = converter.convert(input_pdf)
    
    # 匯出為 Markdown 格式
    markdown_content = result.document.export_to_markdown()
    
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print(f"Docling 轉換完成：{output_md}")

# 執行
convert_with_docling("example.pdf", "output_docling.md")