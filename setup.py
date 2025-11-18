from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="torch-symbolic",
    version="1.1.2",
    author="Liz Tan",
    author_email= "eszt2@cam.ac.uk",
    description="Deep Learning Interpretability with Symbolic Regression.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.11",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch",
        "pysr",
        "numpy",
        "pandas", 
        "scikit-learn",
        "matplotlib",
        "sympy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
