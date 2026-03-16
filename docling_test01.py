import os

from huggingface_hub import snapshot_download

from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import (
    ConversionResult,
    DocumentConverter,
    InputFormat,
    PdfFormatOption,
)


# Source document to convert
source = "./data/ss123.pdf"


det_model_path = "./det_model_path/en_PP-OCRv3_det_infer.onnx"
    
rec_model_path = "./rec_model_path/ch_PP-OCRv4_rec_server_infer.onnx"
 
cls_model_path = "./cls_model_path/ch_ppocr_mobile_v2.0_cls_train.onnx"
   
ocr_options = RapidOcrOptions(
    det_model_path=det_model_path,
    rec_model_path=rec_model_path,
    cls_model_path=cls_model_path,
)

pipeline_options = PdfPipelineOptions(
    ocr_options=ocr_options,
)

# Convert the document
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
        ),
    },
)

conversion_result: ConversionResult = converter.convert(source=source)
doc = conversion_result.document
md = doc.export_to_markdown()

# Create the output filename based on the input filename
input_filename = os.path.basename(source)
input_filename_without_extension = os.path.splitext(input_filename)[0]
output_filename = f"{input_filename_without_extension}.md"

# Create the 'output' directory if it doesn't exist
output_directory = "output"
os.makedirs(output_directory, exist_ok=True)

output_path = os.path.join(output_directory, output_filename)

# Write the Markdown output to the file
with open(output_path, 'w', encoding='utf-8') as outfile:
    outfile.write(md)

print(f"Markdown output written to: {output_path}")