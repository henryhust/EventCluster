'''
Created on 2019年11月9日

@author: Administrator
'''
import sys, os

path = os.path.dirname(os.getcwd())
sys.path.append(path)
from single_pass_v1 import SinglePassV1, Cluster, Document


# 增加倒排索引
class SinglePassV2(SinglePassV1):

    def __init__(self):
        super().__init__()
        self.document_map = {}  # 存储文档信息，id-content结构。当然value也可以使用对象存储文档的其他信息。
        self.cluster_map = {}  # 存储簇的信息，由于需要频繁查询，使用了map ， id-cluster_object结构。
        self.cluster_inverse = {}  # word-cluster_ids结构的倒排索引

    def clustering(self):
        for doc_id in self.document_map:
            words = self.document_map[doc_id].features

            single_flag = True

            for cluster_id in self.get_cand_clusters(words):
                cluster = self.cluster_map[cluster_id]
                if self.similar(cluster, self.document_map[doc_id]):
                    cluster.add_doc(doc_id)
                    single_flag = False
                    break
            if single_flag:
                new_cluster_id = "cluster_" + str(len(self.cluster_map))
                print(new_cluster_id)
                new_cluster = Cluster(new_cluster_id, doc_id)
                self.cluster_map[new_cluster_id] = new_cluster

                for word in self.document_map[new_cluster.center_doc_id].features:
                    if word not in self.cluster_inverse: self.cluster_inverse[word] = []
                    self.cluster_inverse[word].append(new_cluster_id)

    def get_cand_clusters(self, words):
        cand_cluster_ids = []
        for word in words:
            cand_cluster_ids.extend(self.cluster_inverse.get(word, []))
        return cand_cluster_ids

    # 打印所有簇的简要内容
    def show_clusters(self):
        for cluster_id in self.cluster_map:
            cluster = self.cluster_map[cluster_id]
            print(cluster.cluster_id, cluster.center_doc_id, cluster.members)


if __name__ == '__main__':
    docs = ["我爱北京天安门，天安门上太阳升。",
            "我要开着火车去北京，看天安门升旗。",
            "我们的家乡，在希望的田野上。",
            "我的老家是一片充满希望的田野。"]
    single_passor = SinglePassV2()
    single_passor.fit(docs)
    single_passor.show_clusters()
