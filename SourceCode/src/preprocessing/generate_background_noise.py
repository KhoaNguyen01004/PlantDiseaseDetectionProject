import os
import cv2
import numpy as np
import random

base_dir = 'data/raw/color/Background_Noise'
os.makedirs(base_dir, exist_ok=True)

types = ['solid'] * 100 + ['gaussian'] * 200 + ['gradient'] * 200
random.shuffle(types)

for i, img_type in enumerate(types):
    h, w = 224, 224
    if img_type == 'solid':
        val = (i % 3) * 85
        img = np.full((h, w, 3), val, dtype=np.uint8)
    elif img_type == 'gaussian':
        img = np.random.normal(128, 50, (h, w, 3)).clip(0, 255).astype(np.uint8)
    else:  # gradient
        y, x = np.ogrid[:h, :w]
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:,:,0] = (y / h * 255).astype(np.uint8)
        img[:,:,1] = (x / w * 255).astype(np.uint8)
        img[:,:,2] = (128 * (1 + np.sin((x + y) / 50))).astype(np.uint8)
    
    cv2.imwrite(os.path.join(base_dir, f'bg_{i:03d}.png'), img)

print(f'Created 500 Background_Noise images in {base_dir}')

