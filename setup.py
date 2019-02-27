import setuptools

long_description = """`promap` is a command-line tool that you can use
in conjunction with your digital video workflow
to do projection mapping.

`promap` uses a projector and a single camera
to compute the physical scene as viewed from the projector,
as well as a coarse disparity (depth) map.

You can import the maps that it generates straight into your [digital video workflow](https://radiance.video),
or you can post-process them into masks and uvmaps first using image processing tools.
"""

setuptools.setup(
    name="promap",
    version="0.0.2",
    author="Eric Van Albert",
    author_email="eric@van.al",
    description="Projection mapping pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ervanalb/promap",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'scipy',
    ],
    entry_points = {
        'console_scripts': ['promap=promap:main'],
    }
)
