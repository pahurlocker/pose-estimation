from setuptools import setup, find_packages

setup(
    name="pose-estimation",
    version="1.0",
    long_description=__doc__,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "openvino-dev[onnx,tensorflow2]==2021.4.*",
        "matplotlib<3.4",
        "gdown",
        "pytube",
        "click",
        "openvino-extensions",
        "yaspin",
        "Flask",
        "jedi==0.17.2",
        "setuptools>=56.0.0",
        "Pillow>=8.3.2",
        "pygments>=2.7.4",  # not directly required, pinned by Snyk to avoid a vulnerability
        "scikit-learn>=0.24.2",  # not directly required, pinned by Snyk to avoid a vulnerability
        "SQLAlchemy",
        "Flask-SQLAlchemy",
    ],
)
