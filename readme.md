## 面向对话数据的事件聚类
采用依存句法分析对话文本当中的事件要素进行提取，形成SVO实体三元组。并送入BERT或LASER模型进行文本编码，采用SOM或DBSCAN对话题内容进行聚类，以形成特定的事件主题。

# 1.数据准备

/data/corpus.txt为话题数据，涉及AI、历史、金融、电影、科学、体育等方面。

数据来源：
- 对话系统中文语料：https://github.com/candlewill/Dialog_Corpus

# 2.环境准备
2.1 python环境
```
pip install -r requirements.txt
```
2.2 BERT模型
```
百度网盘（提取码wngt）：https://pan.baidu.com/s/1x-jIw1X2yNYHGak2yiq4RQ?pwd=wgnt
```

# 3.项目运行
3.1 运行前提

修改feature/BERT/config下的BERT模型路径

3.2 程序运行
```
python main.py
```