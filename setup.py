from setuptools import setup, find_packages

with open("coords_array/__init__.py", encoding="utf-8") as f:
    line = next(iter(f))
    VERSION = line.strip().split()[-1][1:-1]
      
with open("README.md", "r") as f:
    readme = f.read()

DESC = (
    "N-Dimensional array implemented with slicing, transformation "
    "and interpolation in a coordinate-based manner."
)
setup(
    name="coords-array",
    version=VERSION,
    description=DESC,
    author="Hanjin Liu",
    author_email="liuhanjin-sc@g.ecc.u-tokyo.ac.jp",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="BSD 3-Clause",
    download_url="https://github.com/hanjinliu/coords-array",
    packages=find_packages(exclude=["docs", "tests"]),
    install_requires=[
        "numpy>=1.21",
    ],
    python_requires=">=3.8",
)
