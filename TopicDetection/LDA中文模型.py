#-*- coding: UTF-8 -*-
import jieba, os
from gensim import corpora, models, similarities
train_set = []
i=17910
#walk = os.walk('F:\数据文件-熊\数据集\Sogou\Reduced')
walk = os.walk('Reduced')
for root, dirs, files in walk:
    for name in files:
        f = open(os.path.join(root, name), 'rb')
        raw = f.read()
        i=i-1
        print(i)
        raw=str(raw,encoding='gbk',errors='ignore')
        word_list = list(jieba.cut(raw, cut_all = False))
        # 读取停用词表
        stopword = []
        with open('./stopWord.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                stopword.append(line)
            stopword = stopword[1:]
        word_list = [word for word in word_list if word not in stopword]
        # 去掉空格
        word_list = [word for word in word_list if word.strip() != ""]
        train_set.append(word_list)


dic = corpora.Dictionary(train_set)
corpus = [dic.doc2bow(text) for text in train_set]
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
lda = models.LdaModel(corpus_tfidf, id2word = dic, num_topics = 9,passes=1)
corpus_lda = lda[corpus_tfidf]
# for i in range(0, 9):
#     print(lda.print_topic(i))

print(lda.print_topics(num_topics=9, num_words=3))