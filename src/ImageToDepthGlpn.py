from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation
import torch
import numpy as np
from PIL import Image
from PIL import ImageOps
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

    #load model
    feature_extractor = GLPNFeatureExtractor.from_pretrained("vinvino02/glpn-nyu")
    model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

    # prepare image for the model
    inputs = feature_extractor(images=image, return_tensors="pt")

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
    depth = ImageOps.invert(depth)
    os.chdir(str(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))))
    if not os.path.exists("./imageprocessing"):
        os.makedirs("./imageprocessing")
    depth.save("./imageprocessing/depth.png")
except Exception:
    traceback.print_exc()