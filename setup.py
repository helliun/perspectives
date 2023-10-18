import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="perspectives",
    version="1.0.1",
    author="Henry Leonardi",
    author_email="leonardi.henry@gmail.com",
    description="Extracting directed emotions at scale with LMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/helliun/perspectives",
    packages=["perspectives"],#setuptools.find_packages(),
    #py_modules = ["src/causal_chains"],
    install_requires = ["hatchling","transformers","sentence-transformers","tqdm", "pydot", "graphviz"]
,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)