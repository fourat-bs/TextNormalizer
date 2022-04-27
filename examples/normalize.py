from textnormalizer import TextNormalizer

normalizer = TextNormalizer()

normalized_text = ['senior software engineer', 'solutions architect', 'python developer', 'junior software engineer', 'java developer']
to_normalize = ['experienced software engineer', 'cloud architect', 'entry level software engineer', 'enterprise level software engineer',
                'java programmer']

normalizer.fit(normalized_text)
transformed = normalizer.transform(to_normalize)
print(transformed)
