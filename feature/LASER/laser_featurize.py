import requests


def get_vect(query_in, lang='en', address='101.33.74.244:8050'):
    url = "http://" + address + "/vectorize"
    params = {"q": query_in, "lang": lang}
    resp = requests.get(url=url, params=params).json()
    return resp["embedding"]


def featurize(sentences):
    """基于laser的特征表示"""
    mul_line = "\n".join(sentences)
    print("laser is working")
    return get_vect(mul_line)