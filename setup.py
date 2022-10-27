import os
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = "\n" + f.read()

if __name__ == "__main__":
    setup(
        name="flwr-lowcarb",
        use_scm_version=True,
        author="Philipp Wiesner, Martin Schellenberger, Dennis Grinwald",
        author_email="wiesner@tu-berlin.de",
        description="Carbon-aware client selection strategy for Flower",
        long_description=long_description,
        long_description_content_type='text/markdown',
        keywords=["carbon awareness", "federated learning", "client selection", "flower"],
        url="https://github.com/birnbaum/lowcarb",
        packages=["lowcarb"],
        license="MIT",
        python_requires=">=3.7",
        setup_requires=['setuptools_scm'],
        install_requires=[
            'flower',
            'numpy',
            'pandas',
            "urllib3 >= 1.25.3",  # Carbon Aware SDK client
            "python-dateutil",  # Carbon Aware SDK client
        ],
        classifiers=[
            "Development Status :: 3 - Alpha",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
        ],
    )
