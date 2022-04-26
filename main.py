from textnormalizer import TextNormalizer
import time
import pandas as pd
import os

normalizer = TextNormalizer()
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

normalized_text = ['senior software engineer', 'solutions architect', 'python developer', 'junior software engineer', 'java programmer']
to_normalize = ['experienced software engineer', 'cloud architect', 'entry level software engineer', 'enterprise level software engineer',
                'java developer']

if __name__ == '__main__':
    normalizer.fit(normalized_text)
    normalizer.save('metadata/model')
    model = TextNormalizer.load('metadata/model')
    start = time.time()
    print(model.transform(to_normalize))
    print(time.time() - start)
