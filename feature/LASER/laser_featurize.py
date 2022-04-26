import requests
import numpy as np
from feature.LASER.config import laser_config


class LaserEncoder(object):
    def __init__(self, config):
        self.ip_address = config.get("ip_address")

    def encode(self, query_in, lang='en'):
        """
        基于云服务器搭建的laser向量生成服务
        :param query_in: String, can use '\n' to encode more sentence
        :param lang:
        :return:
        """
        url = f"http://{self.ip_address}/vectorize"
        params = {"q": query_in, "lang": lang}
        resp = requests.get(url=url, params=params).json()
        return np.array(resp["embedding"])

    def featurize(self, sentences):
        """
        基于laser的特征表示
        :param sentences:list of string
        :return:np.array, which shape=[len(sentences), 1024]
        """
        mul_line = "\n".join(sentences)
        print("laser is working")
        return self.encode(mul_line)


if __name__ == '__main__':

    laser_encoder = LaserEncoder(config=laser_config)

    sentences = ["你好我是周杰伦"]
    res = laser_encoder.featurize(sentences)

    print(res.shape)
