from setuptools import setup

setup(
    name="bird",
    version="0.0.0",    
    description="BIRD: behavior induction via representation-structure distillation",
    packages=["bird"],
    install_requires=[
        "einops",
        "fire",
        "numpy",
        "opencv-python",
        "Pillow",
        "scipy",
        "scikit-image",
        "torch",
        "torchvision",
        "tqdm",
        "Wand",
    ],
)