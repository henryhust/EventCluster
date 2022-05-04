import re
import jieba
from stopwords import STOPWORDS
from collections import Counter
import os.path
import jieba.posseg as pseg
from wordcloud import WordCloud
from PIL import Image

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def filter_word(filepath):
    """待过滤词汇"""
    filter_words = set()
    with open(filepath, "r", encoding="utf8") as fr:
        for line in fr.readlines():
            result = re.findall(r"@(.*)@", line)
            if result:
                filter_words.add("@"+result[0])
    return filter_words


def frequency(content):
    """统计文本内容"""

    content_list = [word.strip() for word in jieba.cut(content) if word not in STOPWORDS and len(word) >= 2]

    word2count = Counter(content_list)
    word2count = {key: value for key, value in sorted(word2count.items(), key=lambda x: x[1], reverse=True)[:100]}

    return word2count


def word_cloud(word2count, img_path):
    from matplotlib import colors
    color_list = ['#EE5C42', '#32CD32', '#FFA500', '#1E90FF']  # 建立颜色数组
    colormap = colors.ListedColormap(color_list)  # 调用

    word_cloud_image = WordCloud(font_path=os.path.join(project_path, "word_style/STFANGSO.TTF"),
                                 scale=2,
                                 background_color='white',  collocations=False, max_words=2000, width=960,
                                 height=540, margin=5, colormap=colormap).generate_from_frequencies(word2count)
    image = word_cloud_image.to_image()
    image.show()

    word_cloud_image.to_file(img_path)
    print("词云文件保存在：{}".format(img_path))


if __name__ == '__main__':

    filepath = "../data/lgbd/彩虹后宫_out_co.txt"
    with open(filepath, "r", encoding="utf8") as fr:
        content = fr.read()
        filter_words = filter_word(filepath)
        for word in filter_words:
            content = content.replace(word, "")
        word2count = frequency(content)
        word_pos = pseg.cut(content)

        former_word = ""
        former_pos = ""
        result = set()
        for word, pos in word_pos:
            if pos == former_pos and "n" in pos:
                result.add(former_word+"_"+word)
            former_word = word
            former_pos = pos
        print(result)
        # for key, value in word2count.items():
        #     print(key, value)
        # word_cloud(word2count, img_path="./word_cloud-lgbd.png")

