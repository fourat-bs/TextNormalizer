from setuptools import setup, find_packages

required_packages = [
    "numpy>=1.21.5",
    "torch>=1.11.0",
    "scikit-learn>=1.0.2",
    "transformers>=4.18.0",
    "joblib>=1.1.0"
]

with open("README.md", "r") as f:
    README = f.read()

setup(
    name='t-normalizer',
    version='0.0.1',
    packages=find_packages(exclude=["tests", "examples"]),
    url='https://github.com/fourat-bs/TextNormalizer',
    license='MIT',
    author='Fourat Ben Salah',
    author_email='med.fourat.ben.salah@gmail.com',
    description='TextNormalizer perform fully supervised text normalization',
    long_description=README,
    long_description_content_type="text/markdown",
    python_requires='>=3.7',
    install_requires=required_packages,
)
