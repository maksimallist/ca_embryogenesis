import setuptools

with open("README.md", "r") as fh:  # , encoding="utf-8"
    long_description = fh.read()

setuptools.setup(
    name="neural-cellar-automata",  # Replace with your own username
    version="0.1.0",
    author="Petrov Maksim Andreevich",
    author_email="maksimallist@gmail.com",
    description="An open source library for building and training neural cellar automata.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maksimallist/ca_embryogenesis",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)
