from PIL import Image
import os
for img in os.listdir():
    if ".png" in img:
        png = Image.open(img).convert('RGBA')
        background = Image.new('RGBA', png.size, (255, 255, 255))

        alpha_composite = Image.alpha_composite(background, png)
        alpha_composite.save(img, 'png', quality=80)
