from pathlib import Path
from docling.utils.model_downloader import download_models

models_dir = Path("models")
models_dir.mkdir(parents=True, exist_ok=True)

download_models(
    output_dir=models_dir,
    with_layout=True,
    with_tableformer=True,
    with_easyocr=False,
    with_code_formula=True,
    with_picture_classifier=True,
    progress=True,
)

print(f"Models downloaded to: {models_dir.resolve()}")