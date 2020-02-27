#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : szu-hwj
"""处理数据
"""
import json
import os

import jieba
import pandas as pd


class DataPreproce:
    """
    对数据集进行处理，得到以外格式的数据：
    csv文件：原问题和分词去停用词的问题
    json文件：将问题保存成json文件，为一个list，list中每个元素为一个字典{title:原问题, split_title:切分去停用问题}
    对于每个数据集，只需要输入数据集名和停用词，就都能得到以上两个文件，方便复用
    """

    def __init__(self, csv_filepath, stoplist_filepath, dfdata_filepath, jsondata_filepath):
        self.cwd = os.getcwd()
        self.filename = os.path.basename(csv_filepath)
        self.csv_filepath = csv_filepath
        self.stop_list = self.get_stoplist(stoplist_filepath)
        self.stoplist_filepath = stoplist_filepath
        self.dfdata_filepath = dfdata_filepath
        self.jsondata_filepath = jsondata_filepath

    @staticmethod
    def get_stoplist(stoplist_filepath):
        """
        读取停用词
        :param stoplist_name: 停用词的路径
        :return:
        """
        with open(stoplist_filepath, encoding='utf-8') as f:
            stop_word = f.readlines()
        stop_list = [x.strip() for x in stop_word]
        return stop_list

    def split_word(self, query):
        """
        结巴分词，去除停用词
        :param query: 待切问题
        :param stop_list: 停用词表
        :return:
        """
        words = jieba.cut(query)
        result = ' '.join(
            [word for word in words if word not in self.stop_list])

        # url = 'http://api.pullword.com/get.php?source=' + query + '&param1=1&param2=0'
        # response = requests.get (url)
        # segment_content = response.text
        # segment_list = segment_content.split ('\r\n')
        # result = ' '.join ([word for word in segment_list if word not in self.stop_list])
        # print("result:",result)
        return result

    def split_dataset(self):
        """
        对数据集所有问题进行切词，title列表示未切割问题，split_title表示已经切割
        :return:
        """
        file_path = self.csv_filepath
        f = open(file_path, encoding='utf-8')
        df_dataset = pd.read_csv(f, header=None)
        f.close()
        collumns_name = list(df_dataset.columns)[0]
        df_dataset.rename(columns={collumns_name: 'title'}, inplace=True)
        df_dataset['split_title'] = df_dataset['title'].apply(
            lambda x: self.split_word(x))

        save_filename = 'df_' + self.filename
        df_save_path = os.path.join(
            self.dfdata_filepath, save_filename)  # DataFrame格式的数据存放路径
        df_dataset.to_csv(df_save_path, encoding='utf-8')
        return df_dataset

    def get_dataset(self):
        """
        读取json格式的数据
        :return:
        """
        # 如果路径中存在了json数据就直接读取，若没有就对原数据进行处理
        json_path = os.path.join(
            self.jsondata_filepath, self.filename[:-4] + '.json')
        save_filename = 'df_' + self.filename
        df_save_path = os.path.join(
            self.dfdata_filepath, save_filename)  # DataFrame格式的数据存放路径

        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
        else:
            if os.path.exists(df_save_path):
                f = open(df_save_path, encoding='utf-8')
                df_dataset = pd.read_csv(f)
                f.close()
            else:
                df_dataset = self.split_dataset()
            data = []  # 转变df_dataset的格式与伦青的一致
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
            else:
                for i in range(len(df_dataset)):
                    dict_sample = dict(df_dataset.iloc[i])
                    data.append(dict_sample)
                with open(json_path, 'w') as f:
                    json.dump(data, f)
        return data

    def merge_json(self, insert_filename, total_filename='total_dataset.json'):
        """
        该函数用来将新增的数据加到总的json文件中
        :param insert_filename: 需要加入的数据的json文件
        :param total_filename: 总的数据的json文件
        :return:
        """
        cwd = os.getcwd()
        total_path = cwd + '\\' + total_filename
        insert_path = cwd + '\\' + insert_filename
        with open(insert_path) as f:
            insert_data = json.load(f)
        if not os.path.exists(total_path):
            with open(total_path, 'w') as f:
                json.dump(insert_data, f)
        else:
            with open(total_path) as f:
                total_data = json.load(f)
            total_data.extend(insert_data)
            with open(total_path, 'w') as f:
                json.dump(total_data, f)

# def parse_args():
#     path=os.path.abspath ('..')
#     file_name = 'v_train.csv'
#     stoplist_name = 'stop_words.txt'
#     parser = argparse.ArgumentParser(description='Generate the json file')
#     parser.add_argument('--csvdata_file', help='csvdata_file', default=os.path.join(pathconfig.CSV_DATA_PATH,file_name), type=str)
#     parser.add_argument('--dfdata_filepath', help='dfdata_filepath', default=pathconfig.DF_DATA_PATH, type=str)
#     parser.add_argument('--jsondata_filepath', help='jsondata_filepath', default=pathconfig.JSON_DATA_PATH, type=str)
#     parser.add_argument('--stop_words_file', help='stop_words_file', default=os.path.join(pathconfig.OTHER_DATA_PATH,stoplist_name), type=str)
#
#     args = parser.parse_args()
#     return args
#
# if __name__ == '__main__':
#     start = time.time()
#     #file_name = 'train_questions.csv'
#     args=parse_args()
#     print (args.csvdata_file)
#     print (args.stop_words_file)
#     data = DataPreproce(args.csvdata_file, args.stop_words_file,args.dfdata_filepath,args.jsondata_filepath)
#     data = data.get_dataset()
#     print('spent %.2f times to load data' % (time.time()-start))
#     print(data[0])
#
