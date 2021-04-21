# coding: utf-8

# # Deep Pavlov RuBert for sentense embeddings
# RuBERT was trained on the Russian part of Wikipedia and news data. We used this training data to build vocabulary of
# Russian subtokens and took multilingual version of BERT-base as initialization for RuBERT
# 
# Kuratov, Y., Arkhipov, M. (2019). Adaptation of Deep Bidirectional Multilingual Transformers for Russian Language.
# arXiv preprint arXiv:1905.07213

import numpy as np
from deeppavlov.core.common.file import read_json
from deeppavlov import build_model, configs


class RuBertWrapper:
    def __init__(self, config_path):
        bert_config = read_json(configs.embedder.bert_embedder)
        bert_config['metadata']['variables']['BERT_PATH'] = config_path
        self.rubert_model = build_model(bert_config)

    # Mock question: Что такое искусственный интеллект?
    # input = ['это способность компьютера обучаться, принимать решения
    # и выполнять действия, свойственные человеческому интеллекту',
    # 'это свойство интеллектуальных систем выполнять
    # творческие функции, которые традиционно считаются прерогативой человека']
    # output = distance between sentence embedding vectors

    def calculate_distance(self, texts):
        """ :param texts - list of two string sentences s1 and s2
        :returns distance between mean embeddings for s1 and s2 """
        assert len(texts) == 2

        # Вычисляем эмбеддинги для правильного ответа и ответа кандадата
        tokens, token_embs, subtokens, subtoken_embs, sent_max_embs, sent_mean_embs, bert_pooler_outputs = self.rubert_model(texts)
        # Эмбеддинги имеют размер: (количества слов в предложении, 768)
        # 768 - длина вектора внутреннего состояния модели RuBERT для одного слова
        # Вычисляем среднее значение эмбеддингов по словам в предложении
        e_0 = token_embs[0].mean(axis=0)
        e_1 = token_embs[1].mean(axis=0)
        dist = np.linalg.norm(e_0 - e_1)
        return dist
