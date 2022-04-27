from ._metrics import compute_similarity
from ._exceptions import NotFitedError
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from typing import List
from ._utils import is_valid_input, TNLogger
import os
import joblib

logger = TNLogger('INFO')


class TextNormalizer:
    """
    Create a TextNormalizer model that normalizes the input text to the closest normalized text using Sentence
    Transformers as an encoder.
    :param model_name: The name of the model to use as an encoder, defaults to 'sentence-transformers/all-MiniLM-L6-v2'

    """

    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.sentences = None
        self.embeddings = None

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed(self, sentences: List[str], **kwargs) -> torch.Tensor:
        """
        Encode the sentences using the model's encoder and return the embeddings
        :param sentences: List of strings to encode
        :param kwargs: Additional arguments for the tokenizer
        :return: A tensor with the encoded sentences
        """
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', **kwargs)
        with torch.no_grad():
            output = self.encoder(**encoded_input)
        sentence_embeddings = self.mean_pooling(output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings

    def fit(self, normalized: List[str], **kwargs):
        """
        Compute the embeddings of the normalized sentences and store them in self.embeddings.
        :param normalized: List of strings that are already normalized
        :param kwargs:  Additional arguments for the tokenizer
        :return: A vector representation of the normalized sentences
        """
        is_valid_input(normalized)
        self.embeddings = self.embed(normalized, **kwargs)
        self.sentences = normalized

        return self

    def transform(self, to_normalize: List[str]) -> List[str]:
        """
        Compute the embeddings of the sentences to normalize and return the closest normalized sentence.
        :param to_normalize:
        :return:
        """
        if self.embeddings is None:
            raise NotFitedError(
                f'You need to fit this {type(self).__name__} instance first before using the transform method')
        is_valid_input(to_normalize)
        to_normalize_embeddings = self.embed(to_normalize)
        indices = compute_similarity(to_normalize_embeddings, self.embeddings)
        return [self.sentences[i] for i in indices]

    def save(self, path: str):
        """
        Save the model to a file
        :param path: a valid path
        """
        if self.embeddings is None:
            logger.info('attempt to save a model that has not been fitted yet')
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'wb') as f:
            logger.info(f'Saving model to {path}')
            joblib.dump(self, f)

    @classmethod
    def load(cls, path: str):
        """
        Load a model from a file
        :param path: valid path of existing file
        :return: instance of TextNormalizer
        """
        if not os.path.exists(os.path.abspath(path)):
            raise Exception(f'File path ({path}) does not exist!')
        with open(path, 'rb') as f:
            logger.info(f'Loading model from {path}')
            model = joblib.load(f)
        return model
