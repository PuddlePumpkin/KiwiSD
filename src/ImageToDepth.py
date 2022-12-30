from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests
import os
from io import BytesIO
import sys
import traceback
try:
    url = str(sys.argv[1])
    print("Generating depth from url: "+url)
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

    # prepare image for the model
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).clamp(min=0, max=255)

    # visualize the prediction
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    os.chdir(str(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))))
    if not os.path.exists("./imageprocessing"):
        os.makedirs("./imageprocessing")
    depth.save("./imageprocessing/depth.png")
except Exception:
    traceback.print_exc()