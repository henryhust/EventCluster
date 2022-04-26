from feature.LASER import laser_featurize
from EventExtraction.baidu_svo_extract import SVOParser

if __name__ == '__main__':
    sentences = ["今天天气非常好", "我能过通过这场考试我觉得很开心", "我想没有什么事情是解决不了的"]
    vecs = laser_featurize.featurize(sentences)
    print(vecs)