from pathlib import Path
import os
from io import BytesIO
from PIL import Image
from PIL import ImageOps
os.chdir(str(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))))
filelist = list(Path("./results/").rglob("*.png"))

#for file in filelist:
def png_to_jpeg(png_file_path:Path):
    print(png_file_path)
    with Image.open(png_file_path) as img:
        file_name = png_file_path.stem
        img.save(f'./jpegresults/{file_name}.jpeg', 'JPEG', quality = 90)

for file in filelist:
    png_to_jpeg(file)