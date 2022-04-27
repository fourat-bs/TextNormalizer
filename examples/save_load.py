from textnormalizer import TextNormalizer

normalizer = TextNormalizer()
model_path = 'models/my_model'

# fit and save the model
normalized_text = ['senior software engineer', 'solutions architect', 'python developer', 'junior software engineer', 'java programmer']
normalizer.fit(normalized_text)
normalizer.save(model_path)

# load the model and use it for inference
to_normalize = ['experienced software engineer', 'cloud architect', 'entry level software engineer', 'enterprise level software engineer',
                'java developer']
model = TextNormalizer.load(model_path)
transformed = model.transform(to_normalize)
