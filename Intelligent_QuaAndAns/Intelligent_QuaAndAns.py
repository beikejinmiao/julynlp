import argparse
import json
import sys as _sys
from time import time as t
import shutil
import pandas as pd

from Config.allconfig import *
from Filter_sentence import (get_same_rate, length_difference_rate,
                             load_target_sentence)
from Model.ES_model import ES_Model
from Model.TfIdf_model import Tf_Idf_Model
from Preproce_Data.Data import DataPreproce


class Intelligent_QuaAndAns(object):
    def __init__(self, args):
        self.config = args
        self.context_file = self.split_idAndcontext(args.context_file)

        # self.context_file=args.context_file
        self.jsondata_file = args.jsondata_file
        self.stop_words_file = args.stop_words_file
        self.dfdata_filepath = args.dfdata_filepath

        self.dfdata_file = os.path.join(
            self.dfdata_filepath, "df_" + os.path.basename(self.context_file))
        print("self.dfdata_file:", self.dfdata_file)

        self.dictionary_file = args.dictionary_file

        self.input_sentence = args.input_sentence
        self.topn = args.topn
        self.length_difference_rate = args.length_difference_rate
        self.same_rate = args.same_rate
        self.simsmodel_name = args.simsmodel_name
        self.wmdrate = args.wmdrate

        self.index_name = args.index_name
        self.doc_type_name = args.doc_type_name
        self.index = args.index

        if os.path.exists(self.jsondata_file):
            with open(self.jsondata_file, 'r') as f:
                self.contextdata = json.load(f)
        else:
            data = DataPreproce(self.context_file, self.stop_words_file,
                                self.dfdata_filepath, os.path.dirname(self.jsondata_file))
            data.get_dataset()
            with open(self.jsondata_file, 'r') as f:
                self.contextdata = json.load(f)

        self.bulid_model()

    def split_idAndcontext(self, context_file):
        newcontext_file = context_file.split(".")[0] + "_context.csv"
        id_file = context_file.split(".")[0] + "_id.csv"
        self.id_file = id_file

        if os.path.exists(newcontext_file):
            return newcontext_file
        else:
            df = pd.read_csv(context_file, delimiter='\t', header=None)
            df.columns = ["id", "context"]
            id = df["id"]
            context = df["context"]
            id.to_csv(id_file, header=None, index=False, encoding='utf-8')
            context.to_csv(newcontext_file, header=None,
                           index=False, encoding='utf-8')
            return newcontext_file

    def bulid_model(self):
        if self.simsmodel_name == 'tfidf':
            self.model = Tf_Idf_Model(
                self.contextdata, self.dictionary_file, self.stop_words_file)
        elif self.simsmodel_name == 'es':
            self.model = ES_Model(self.dfdata_file, self.index_name, self.doc_type_name, self.stop_words_file,
                                  self.index)
        else:
            print('不存在这个模型')
            return "不存在这个模型"

    def get_topn_sims(self):
        # start_time = t()
        results = self.model.get_topn_sims(self.input_sentence, self.topn)

        # tf_data=results.copy()
        # self.ex_model = Tf_Idf_Model(
        #     self.contextdata, self.dictionary_file, self.stop_words_file,'tfidf',tf_data)

        target_sentence, target_key = load_target_sentence(results, self.topn)
        print("target_key", target_key)
        confidence = True  # 置信度

        if target_sentence != 0:
            len_diff_rate = length_difference_rate(
                self.input_sentence, target_sentence)
            if len_diff_rate > self.length_difference_rate:  # 两个问题长度差100%，这个可以调整
                confidence = False
            else:
                same_rate = get_same_rate(
                    self.input_sentence, target_sentence, self.model.stop_list)
                if same_rate < self.same_rate:
                    confidence = False

                    # 找到对应的要筛选的句子,形如{'index':编号,'similarity':相似度，'title':问题}
        if target_key > 0:
            target_result_2 = results['result2'][target_key]
            if not confidence:
                target_result_2['confidence'] = '不准确'
            else:
                target_result_2['confidence'] = '准确'

        df = pd.read_csv(self.id_file, delimiter='\t', header=None)
        id = df.iloc[int(results['result2'][0]['index']) - 1][0]

        for context_id in results['result2']:
            results['result2'][context_id]['index'] = str(df.iloc[int(results['result2'][context_id]['index']) - 1][0])

        print("id:", id)
        # results_string = json.dumps (results)
        # print("cost time:", self.get_time_dif(start_time))
        return id, results

    def get_time_dif(self, start_time):
        end_time = t()
        time_dif = end_time - start_time
        return time_dif

    def get_sentence_category(self):
        return "cate"


def parse_args(context_file_name=None, stoplist_name=None, input_sentence=None, simsmodel_name=None, topn=None,
               is_build=True):
    use_id = context_file_name.split('_')[0]
    newcontext_file = context_file_name.split(".")[0] + "_context.csv"
    print("use_id:", use_id)

    if is_build:
        print("jjjj")
        if os.path.exists(os.path.join(pathconfig.OTHER_DATA_PATH, use_id)):
            shutil.rmtree(os.path.join(pathconfig.OTHER_DATA_PATH, use_id))
            os.makedirs(os.path.join(pathconfig.OTHER_DATA_PATH, use_id))
        else:
            os.makedirs(os.path.join(pathconfig.OTHER_DATA_PATH, use_id))
        if os.path.exists(os.path.join(pathconfig.JSON_DATA_PATH, use_id)):
            shutil.rmtree(os.path.join(pathconfig.JSON_DATA_PATH, use_id))
            os.makedirs(os.path.join(pathconfig.JSON_DATA_PATH, use_id))
        else:
            os.makedirs(os.path.join(pathconfig.JSON_DATA_PATH, use_id))
        if os.path.exists(os.path.join(pathconfig.DF_DATA_PATH, use_id)):
            shutil.rmtree(os.path.join(pathconfig.DF_DATA_PATH, use_id))
            os.makedirs(os.path.join(pathconfig.DF_DATA_PATH, use_id))
        else:
            os.makedirs(os.path.join(pathconfig.DF_DATA_PATH, use_id))
    else:
        print("ssss")
        if not os.path.exists(os.path.join(pathconfig.CSV_DATA_PATH, use_id)):
            os.makedirs(os.path.join(pathconfig.CSV_DATA_PATH, use_id))
        if not os.path.exists(os.path.join(pathconfig.OTHER_DATA_PATH, use_id)):
            os.makedirs(os.path.join(pathconfig.OTHER_DATA_PATH, use_id))
        if not os.path.exists(os.path.join(pathconfig.JSON_DATA_PATH, use_id)):
            os.makedirs(os.path.join(pathconfig.JSON_DATA_PATH, use_id))
        if not os.path.exists(os.path.join(pathconfig.DF_DATA_PATH, use_id)):
            os.makedirs(os.path.join(pathconfig.DF_DATA_PATH, use_id))

    parser = argparse.ArgumentParser(description='Generate the json file')
    parser.add_argument('--context_file', help='context_file',
                        default=os.path.join(os.path.join(pathconfig.CSV_DATA_PATH, use_id), (
                            filename_Config.context_file_name if context_file_name == None else context_file_name)),
                        type=str)
    parser.add_argument('--stop_words_file', help='stop_words_file',
                        default=os.path.join(pathconfig.OTHER_DATA_PATH, (
                            filename_Config.stoplist_name if stoplist_name == None else stoplist_name)), type=str)
    parser.add_argument('--dictionary_file', help='dictionary_file',
                        default=os.path.join(os.path.join(pathconfig.OTHER_DATA_PATH, use_id), (
                            filename_Config.context_file_name if newcontext_file == None else newcontext_file).split(
                            ".")[0] + ".dict"), type=str)
    parser.add_argument('--jsondata_file', help='jsondata_file',
                        default=os.path.join(os.path.join(pathconfig.JSON_DATA_PATH, use_id), (
                            filename_Config.context_file_name if newcontext_file == None else newcontext_file).split(
                            ".")[0] + ".json"), type=str)
    parser.add_argument('--dfdata_filepath', help='dfdata_filepath',
                        default=os.path.join(pathconfig.DF_DATA_PATH, use_id), type=str)

    parser.add_argument('--index_name', help='index_name',
                        default=str(use_id), type=str)

    parser.add_argument('--min_count', help='min_count',
                        default=default_d2v_trainConfig.Min_Count, type=int)
    parser.add_argument('--windowsize', help='windowsize',
                        default=default_d2v_trainConfig.Window_Size, type=int)
    parser.add_argument('--size', help='size',
                        default=default_d2v_trainConfig.Size, type=int)
    parser.add_argument('--workers', help='workers',
                        default=default_d2v_trainConfig.Workers, type=int)
    parser.add_argument('--epoch_num', help='epoch_num',
                        default=default_d2v_trainConfig.Epoch_Num, type=int)

    parser.add_argument('--input_sentence', help='input_sentence', default=input_sentence,
                        type=str)
    parser.add_argument('--simsmodel_name', help='which model get sims ', default=(
        default_simConfig.simsmodel_name if simsmodel_name == None else simsmodel_name), type=str)
    parser.add_argument('--topn', help='topn',
                        default=(default_simConfig.Topn if topn == None else topn), type=int)
    parser.add_argument('--wmdrate', help='wmdrate',
                        default=default_simConfig.WmdRate, type=int)

    parser.add_argument('--length_difference_rate', help='length_difference_rate',
                        default=default_filterConfig.Length_Dif_Rate, type=float)
    parser.add_argument('--same_rate', help='same_rate',
                        default=default_filterConfig.Same_Rate, type=float)

    parser.add_argument('--doc_type_name',
                        help='doc_type_name', default="context", type=str)
    parser.add_argument('--index', help='index', default=True, type=bool)

    args = parser.parse_args(args=_sys.argv[3:])
    return args


def ThreadRun(is_build, context_file_name=None, stoplist_name=None, input_sentence='请问下物流公司税负率多少？老师',
              simsmodel_name=None, topn=5):
    start_time = t()
    if is_build:
        print("build model")
        args = parse_args(context_file_name, stoplist_name,
                          input_sentence, simsmodel_name, topn, is_build)
        if (args == -1):
            return 0
        QuaAndAns = Intelligent_QuaAndAns(args)
        print("cost time:", t() - start_time)
    else:
        args = parse_args(context_file_name, stoplist_name,
                          input_sentence, simsmodel_name, topn, is_build)
        if (args == -1):
            return 0
        args.index = False
        QuaAndAns = Intelligent_QuaAndAns(args)
        # 求相似度
        id, results = QuaAndAns.get_topn_sims()
        print(results)
        print("cost time:", t() - start_time)
        return id, results


if __name__ == '__main__':
    ThreadRun(False, context_file_name='1_2018-08-14-18-43-11.csv', stoplist_name='stop_words.txt', simsmodel_name='es')
