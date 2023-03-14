import setuptools

with open("requirements.txt") as f:
    required = f.read().splitlines()

setuptools.setup(
    name="knowledge",
    version="0.0.1",
    author="Raphael Sourty",
    author_email="raphael.sourty@gmail.com",
    description="Database",
    url="https://github.com/raphaelsty/knowledge",
    packages=setuptools.find_packages(),
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
