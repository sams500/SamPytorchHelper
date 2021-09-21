from setuptools import setup

setup(
    name="SamsPytorchHelper",  # the pip install name
    version= "0.0.1",
    description="Make pytorch implementation easy so that we one focus on what is really important!!!!",
    py_modules=["TorchHelper"],  # the import name module
    package_dir={'': 'src'}  # '' represent the root directory which is src
)
