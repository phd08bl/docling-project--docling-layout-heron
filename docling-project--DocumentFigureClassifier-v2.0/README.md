---
license: apache-2.0
base_model:
- google/efficientnet-b0
---


# EfficientNet-B0 Document Image Classifier

This is an image classification model based on **Google EfficientNet-B0**, fine-tuned to classify input images into one of the following 26 categories:

1. **logo**
2. **photograph**
3. **icon**
4. **engineering_drawing**
5. **line_chart**
6. **bar_chart**
7. **other**
8. **table**
9. **flow_chart**
10. **screenshot_from_computer**
11. **signature**
12. **screenshot_from_manual**
13. **geographical_map**
14. **pie_chart**
15. **page_thumbnail**
16. **stamp**
17. **music**
18. **calendar**
19. **qr_code**
20. **bar_code**
21. **full_page_image**
22. **scatter_plot**
23. **chemistry_structure**
24. **topographical_map**
25. **crossword_puzzle**
26. **box_plot**



### How to use - Transformers
Example of how to classify an image into one of the 26 classes using transformers:

```python
import torch
import torchvision.transforms as transforms

from transformers import EfficientNetForImageClassification
from PIL import Image
import requests


urls = [
    'http://images.cocodataset.org/val2017/000000039769.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000001750.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000000001.jpg'
]

image_processor = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.47853944, 0.4732864, 0.47434163],
        ),
    ]
)

images = []
for url in urls:
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    image = image_processor(image)
    images.append(image)


model_id = 'docling-project/DocumentFigureClassifier-v2.0'

model = EfficientNetForImageClassification.from_pretrained(model_id)

labels = model.config.id2label

device = torch.device("cpu")

torch_images = torch.stack(images).to(device)

with torch.no_grad():
    logits = model(torch_images).logits  # (batch_size, num_classes)
    probs_batch = logits.softmax(dim=1)  # (batch_size, num_classes)
    probs_batch = probs_batch.cpu().numpy().tolist()

for idx, probs_image in enumerate(probs_batch):
    preds = [(labels[i], prob) for i, prob in enumerate(probs_image)]
    preds.sort(key=lambda t: t[1], reverse=True)
    print(f"{idx}: {preds}")
```


### How to use - ONNX
Example of how to classify an image into one of the 26 classes using onnx runtime:

```python
import onnxruntime

import numpy as np
import torchvision.transforms as transforms

from PIL import Image
import requests

LABELS = [
    "logo",
    "photograph",
    "icon",
    "engineering_drawing",
    "line_chart",
    "bar_chart",
    "other",
    "table",
    "flow_chart",
    "screenshot_from_computer",
    "signature",
    "screenshot_from_manual",
    "geographical_map",
    "pie_chart",
    "page_thumbnail",
    "stamp",
    "music",
    "calendar",
    "qr_code",
    "bar_code",
    "full_page_image",
    "scatter_plot",
    "chemistry_structure",
    "topographical_map",
    "crossword_puzzle",
    "box_plot"
]


urls = [
    'http://images.cocodataset.org/val2017/000000039769.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000001750.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000000001.jpg'
]

images = []
for url in urls:
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    images.append(image)


image_processor = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.47853944, 0.4732864, 0.47434163],
        ),
    ]
)


processed_images_onnx = [image_processor(image).unsqueeze(0) for image in images]

# onnx needs numpy as input
onnx_inputs = [item.numpy(force=True) for item in processed_images_onnx]

# pack into a batch
onnx_inputs = np.concatenate(onnx_inputs, axis=0)

ort_session = onnxruntime.InferenceSession(
    "./DocumentFigureClassifier-v2_0-onnx/model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)


for item in ort_session.run(None, {'input': onnx_inputs}):
    for x in iter(item):
        pred = x.argmax()
        print(LABELS[pred])
```



## Citation
If you use this model in your work, please cite the following papers:

```
@article{Tan2019EfficientNetRM,
  title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
  author={Mingxing Tan and Quoc V. Le},
  journal={ArXiv},
  year={2019},
  volume={abs/1905.11946}
}

@techreport{Docling,
  author = {Deep Search Team},
  month = {8},
  title = {{Docling Technical Report}},
  url={https://arxiv.org/abs/2408.09869},
  eprint={2408.09869},
  doi = "10.48550/arXiv.2408.09869",
  version = {1.0.0},
  year = {2024}
}
```