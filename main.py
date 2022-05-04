# encoding=utf8
import joblib
import logging
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def get_corpus(filepath):
    """读取数据文件"""
    with open(filepath, "r", encoding="utf8") as fr:
        return [line.strip() for line in fr.readlines()]


if __name__ == '__main__':

    data_filepath = "./data/corpus.txt"             # 话题语料文件

    triple_savepath = "./param/triple.list"         # 三元组存储位置
    vec_savepath = "./param/corpus.vec"             # 文档向量存储位置

    reuse_flag = True      # 文档向量重用标识

    if not reuse_flag:

        from EventExtraction.baidu_svo_extract import SVOParser
        from feature.BERT.bert_featurize import BertEncoder
        from feature.BERT.config import bert_config

        sentences = get_corpus(filepath=data_filepath)

        triple_paser = SVOParser()
        triples = triple_paser.extract_triple(sentences)

        logging.info("Bert编码中...")
        bert_encoder = BertEncoder(config=bert_config)
        X = bert_encoder.featurize(sentences)

        logging.info(f"三元组存储位置：{triple_savepath}")
        joblib.dump(triples, triple_savepath)

        logging.info(f"文档向量存储位置：{vec_savepath}")
        joblib.dump(X, vec_savepath)
    else:
        logging.info("load三元组")
        triples = joblib.load(triple_savepath)

        logging.info("load文本向量")
        X = joblib.load(vec_savepath)

    X = np.squeeze(X, axis=1)

    logging.info("聚类中...")
    # cluster = DBSCAN(eps=0.3, min_samples=10, metric="cosine")
    # cluster.fit(X)

    km = KMeans(n_clusters=5)
    pca = PCA(n_components=2)

    vectors_ = pca.fit_transform(X)
    print(vectors_)
    y_ = km.fit_predict(vectors_)  # 聚类
    print(y_)


