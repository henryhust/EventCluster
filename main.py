from feature.LASER.config import laser_config
from feature.BERT.config import bert_config
from feature.LASER.laser_featurize import LaserEncoder
from feature.BERT.bert_featurize import BertEncoder
from EventExtraction.baidu_svo_extract import SVOParser


from sklearn.cluster import DBSCAN, AgglomerativeClustering

if __name__ == '__main__':
    sentences = ["今天天气非常好", "我能过通过这场考试我觉得很开心", "我想没有什么事情是解决不了的"]

    # 三元组提取
    triple_paser = SVOParser()
    triples = triple_paser.extract_triple(sentences)

    # 特征表示
    laser_encoder = LaserEncoder(config=laser_config)
    X = laser_encoder.featurize(sentences)

    # bert_encoder = BertEncoder(config=bert_config)
    # X = bert_encoder.featurize(sentences)

    # 聚类
    cluster = DBSCAN(eps=0.3, min_samples=10)
    cluster.fit(X)
    print(cluster.labels_)
