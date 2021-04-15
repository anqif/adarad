from setuptools import setup

setup(
	name = "adarad",
	version = "0.0.1",
	author = "Anqi Fu",
	author_email = "anqif@stanford.edu",
	packages = ["adarad",],
	license = "Creative Commons Attribution-Noncommercial-Share Alike license",
	long_description = open("README.md").read(),
	long_description_content_type = "text/markdown",
	url = "https://github.com/anqif/adarad",
	install_requires = ["cvxpy>=1.0",
			    "matplotlib",
			    "numpy>=1.15",
			    "pyyaml"],
	python_requires = ">=3.6",
)

