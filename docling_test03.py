from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# Local paths
pdf_path = Path(r"D:\PythonProject\document-agent\data\ss123.pdf")
artifacts_path = Path(r"D:\PythonProject\document-agent\models")
output_md = Path(r"D:\PythonProject\document-agent\sample.md")

# Tell Docling to use only local model artifacts
pipeline_options = PdfPipelineOptions(
    artifacts_path=str(artifacts_path)
)

# Create converter for PDF
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)

# Convert local PDF
result = converter.convert(str(pdf_path))
doc = result.document

# Export to Markdown
markdown_text = doc.export_to_markdown()
output_md.write_text(markdown_text, encoding="utf-8")

print("Done.")
print(f"Markdown written to: {output_md}")