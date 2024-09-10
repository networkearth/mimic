from setuptools import setup, find_packages

setup(
    name="mimic",
    version="0.0.1",
    author="Marcel Gietzmann-Sanders",
    author_email="marcelsanders96@gmail.com",
    packages=find_packages(include=["mimic", "mimic*"]),
    install_requires=[
        "click==8.1.7",
        "tensorflow==2.16.1",
        "pandas==2.2.2",
    ],
    entry_points={
        "console_scripts": [
            "mimic = mimic.cli:cli",
        ]
    },
)