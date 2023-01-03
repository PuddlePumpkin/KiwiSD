import torch
from PIL import Image
import torch
from PIL import ImageOps
import requests
import os
from io import BytesIO
import traceback
import sys
from RealESRGAN import RealESRGAN
try:
    url = str(sys.argv[1])
    print("ESRGAN upscaling from url: " + url)
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    if (image.width > 1024):
        image = image.resize((1024,int(image.height/(image.width/1024))),Image.Resampling.LANCZOS)
    if (image.height > 1024):
        image = image.resize((int(image.width/(image.height/1024)),1024),Image.Resampling.LANCZOS)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.chdir(str(os.path.abspath(os.path.dirname(__file__))))
    model = RealESRGAN(device, scale=2)
    #https://huggingface.co/sberbank-ai/Real-ESRGAN/tree/main
    model.load_weights('weights/RealESRGAN_x2.pth', download=True)
    sr_image = model.predict(image)

    os.chdir(str(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))))
    if not os.path.exists("./imageprocessing"):
        os.makedirs("./imageprocessing")
    sr_image.save("./imageprocessing/upscaled.png")
    file_stats = os.stat("./imageprocessing/upscaled.png")
    if ((file_stats.st_size / (1024 * 1024)) >= 8):
        image = Image.open("./imageprocessing/upscaled.png")
        image.save("./imageprocessing/upscaled.png", 'JPEG', quality = 95)

except Exception:
    traceback.print_exc()