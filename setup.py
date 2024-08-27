from setuptools import setup, find_packages

setup(
    name="cmiou_metric",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "torch",
        "torchvision",
        "Pillow",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for calculating Concept-calibrated Mean IoU (CMIoU) for open-vocabulary semantic segmentation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cmiou_metric",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)