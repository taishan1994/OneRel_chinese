from keras_bert import Tokenizer
import codecs
import unicodedata
try:
    from .tokenization import BasicTokenizer
except Exception as e:
    from tokenization import BasicTokenizer
from transformers import BertTokenizer

"""
The HBTokenizer is critical to OneRel.
"""
basicTokenizer = BasicTokenizer(do_lower_case=True)
try:
    tokenizer = BertTokenizer.from_pretrained('./pre_trained_bert/vocab.txt')
except Exception as e:
    tokenizer = BertTokenizer.from_pretrained('../pre_trained_bert/vocab.txt')

class HBTokenizer(Tokenizer):
    def _tokenize(self, text):
        if not self._cased:
            text = unicodedata.normalize('NFD', text)
            text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
            # text = text.lower()
        spaced = ''
        for ch in text:
            if ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch
        tokens = []
        spaced = basicTokenizer.tokenize(spaced)
        for word in spaced:
            # tokens += self._word_piece_tokenize(word)
            tokens += tokenizer.tokenize(word)
            tokens.append('[unused1]')
        return tokens


def get_tokenizer(vocab_path):
    token_dict = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return HBTokenizer(token_dict, cased=True)


if __name__ == '__main__':
    text = '你好jack chan呀'
    vocab_path = '../pre_trained_bert/vocab.txt'
    tokenier = get_tokenizer(vocab_path)
    print(tokenier.tokenize(text))
