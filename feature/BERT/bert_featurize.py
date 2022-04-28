import logging
import numpy as np

from tqdm import tqdm
from feature.BERT.config import bert_config
from sklearn.metrics.pairwise import cosine_similarity

from bert4keras.snippets import to_array
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model


class BertEncoder(object):
    def __init__(self, config):
        logging.info("Bert参数初始化")
        self.bert = build_transformer_model(
            config_path=config.get("config_path"),
            checkpoint_path=config.get("checkpoint_path"),
            model="bert")

        self.tokenizer = Tokenizer(config.get("dict_path"), do_lower_case=True)

    def tokenize(self, sentence):
        """token2id"""
        token_ids, segments_ids = self.tokenizer.encode(sentence)
        token_ids, segments_ids = to_array([token_ids], [segments_ids])
        return token_ids, segments_ids

    def encode(self, sentence, mod="cls"):
        """
        对句子进行编码
        :param sentence:String
        :param mod: 选择句子编码模式，cls or mean，表示直接取cls向量 或 取词向量平均
        :return: 句子向量
        """
        token_ids, segments_ids = self.tokenize(sentence)
        result = self.bert.predict([token_ids, segments_ids])

        if mod == "cls":
            return result[:, 0, :]
        else:
            return np.sum(result[:, 1:len(token_ids)+2, :], axis=1)/(len(token_ids)-2)

    def featurize(self, sentences):
        return np.array([self.encode(sentence) for sentence in tqdm(sentences)])


if __name__ == '__main__':

    bert_encoder = BertEncoder(config=bert_config)

    sentence_emb1 = bert_encoder.encode("上海自来水来自海上", mod="mean")
    sentence_emb2 = bert_encoder.encode("海上水自来上海来自", mod="mean")

    print(sentence_emb1.shape)
    print(sentence_emb2.shape)
    score = cosine_similarity(X=sentence_emb1, Y=sentence_emb2)
    print(score)


    result = bert_encoder.featurize(sentences=["上海自来水来自海上", "海上水自来上海来自"])
    result = np.squeeze(result, axis=1)

    print(result.shape)

