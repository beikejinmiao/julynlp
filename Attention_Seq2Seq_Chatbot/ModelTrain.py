#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Attention_Seq2Seq_Chatbot.model.nnModel import *
from Attention_Seq2Seq_Chatbot.model.corpusSolver import *
import torch

# load the data
dataClass = Corpus('./corpus/qingyun.tsv', maxSentenceWordsNum=25)

# train model
model = Seq2Seq(dataClass, featureSize=256, hiddenSize=256,
                attnType='L', attnMethod='general',
                encoderNumLayers=3, decoderNumLayers=2,
                encoderBidirectional=True,
                device=torch.device('cuda:0'))
model.train(batchSize=1024, epoch=500)


# save model
model.save('model.pkl')

# bulid a chatbot by using the model
chatbot = ChatBot('model.pkl')


# generate the answer by using greedy search
chatbot.predictByGreedySearch("你好啊")

# generate the answer by using beam search
chatbot.predictByBeamSearch("什么是ai", isRandomChoose=True, beamWidth=10)