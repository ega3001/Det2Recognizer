import setuptools

with open("README.md") as file:
    read_me_description = file.read()

setuptools.setup(
    name="Det2Rec",
    version="0.1",
    author="egor.bakharev",
    author_email="progr18@pancir.it",
    description="This is a Detectron2 wrapper",
    long_description=read_me_description,
    install_requires=[
        "detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.6"
    ],
    long_description_content_type="text/markdown",
    url="https://git.pancir.it/egor.bakharev/SharedLib-Det2Rec",
    packages=['Det2Rec'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
