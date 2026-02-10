from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# 設定 OCR 引擎為 RapidOCR
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.ocr_options = RapidOcrOptions() # 指定使用 RapidOCR

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

# 轉換您提供的範例文件
result_rapid = converter.convert("sample_table.pdf")

print("--- RapidOCR Output ---")
print(result_rapid.document.export_to_markdown())