import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setuptools.setup(
    name="SamPytorchHelper",  # the pip install name
    version="0.0.1",
    author="Soilihi Abderemane",
    author_email="Abderemane500@gmail.com",
    description="Make pytorch implementation easy, so that one can focus on what is really important!!!!",
    long_description=long_description,
    long_description_content_type="text/markdown",

    package_dir={'': 'src'},  # '' represent the root directory which is src
    url="https://github.com/sams500/SamPytorchHelper",
    project_urls={
        "Bug Tracker": "https://github.com/sams500/SamPytorchHelper/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
