from pathlib import Path
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent  # this will give the path where this sript is living 

image_path = SCRIPT_DIR.parent / "output.png"   # it tells to go one-folder UP as the .parent is being used 

try:
    with Image.open(image_path) as img:
        width, height = img.size
    print(f"Width: {width}px")
    print(f"Height: {height}px")
except FileNotFoundError:
    print("Error: Still could not find the file. Check the path printed above!")