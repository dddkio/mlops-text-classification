from setuptools import setup, find_packages

setup(
    name="mlops-text-classification",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "fastapi",
        "pandas",
        "numpy",
        "scikit-learn",
        "pytest"
    ]
)