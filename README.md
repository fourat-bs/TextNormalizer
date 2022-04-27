[![PyPI - Python](https://img.shields.io/badge/python-v3.7+-blue.svg)](https://pypi.org/project/bertopic/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# TextNormalizer
`TextNormalizer` is a strings normalizer that uses [SentenceTransformers](https://www.sbert.net/) as a backbone to obtain vector representations of sentences.
It is designed for repeated normalization of strings against a large corpus of strings. <br>
The main contribution of `TextNormalizer` is to gain time by eliminating the need to compute the normalized strings embeddings every time. 
## Setup
``` bash
pip install t-normalizer
```
## Usage
1. Create and instance of `TextNormalizer`, can be initialized with a `SentenceTransformer` model or a `SentenceTransformer` model path.
2. Obtain the vector representation of the normalized string with `.fit` method.
3. Transform the string with to the most similar normalized form using the `.transform` method.

``` python
from textnormalizer import TextNormalizer

normalizer = TextNormalizer()

normalized_text = ['senior software engineer', 'solutions architect', 'junior software developer']
to_normalize = ['experienced software engineer', 'software architect', 'entry level software engineer']

normalizer.fit(normalized_text)
transformed = normalizer.transform(to_normalize)
```
The model along with the normalized strings and their  vector representations can be saved and loaded with `.save` and `.load` methods.
## Serialization

``` python
# save
normalizer.save('path/to/model')

# load
model = TextNormalizer.load('path/to/model')
```