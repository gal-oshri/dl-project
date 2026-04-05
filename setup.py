from setuptools import setup, find_packages

setup(
    name="dl_project",
    version="0.1.0",
    description="Compressed Context Memory for Dialogue Summarization",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0,<3.0",
        "peft>=0.6.0",
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.0",
        "sentencepiece>=0.1.99",
        "huggingface_hub>=0.19.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "ccm-train=dl_project.train:main",
            "ccm-infer=dl_project.infer:main",
        ],
    },
)
