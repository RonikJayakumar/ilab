step 1: git clone
step 2: run requirement
step 3: Change the IP address in server.py and client.py
step 4: ./run.sh


Potential improvements
Use of DALI - Resizing, augmenting, and training can be done using the GPU for Large datasets.


Preprocessing & Augmentation: CT scans require specific preprocessing steps (e.g., normalization, windowing for lung tissue, etc.) and augmentations like rotations or flips. Use torchvision.transforms or albumentations for more medically relevant augmentations.

import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.Resize(256, 256),  # Resize to a larger size for medical data
    A.Normalize(mean=(0.5,), std=(0.5,)),  # Normalize appropriately for CT scans
    A.RandomRotate90(),  # Medical data might require different types of augmentations
    ToTensorV2(),
])








