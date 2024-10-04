import versioneer
from setuptools import find_packages, setup

setup(
    name="Pyrphoros",
    version=versioneer.get_version(),  # type: ignore
    cmdclass=versioneer.get_cmdclass(),  # type: ignore
    packages=find_packages(),
    install_requires=["varname"],
)
