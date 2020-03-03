"""
配置文件
"""
import os
from easydict import EasyDict as edict


MAIN_HOME = os.path.abspath(os.path.dirname(__file__))  # 返回项目主入口文件的路径


print("base_path: ", MAIN_HOME)
# 路径相关参数
pathconfig = edict()
pathconfig.base_path = MAIN_HOME
pathconfig.DF_DATA_PATH = os.path.join(
    os.path.join(MAIN_HOME, 'Data'), 'df_Data')
pathconfig.CSV_DATA_PATH = os.path.join(
    os.path.join(MAIN_HOME, 'Data'), 'csv_Data')
pathconfig.JSON_DATA_PATH = os.path.join(
    os.path.join(MAIN_HOME, 'Data'), 'json_Data')
pathconfig.OTHER_DATA_PATH = os.path.join(
    os.path.join(MAIN_HOME, 'Data'), 'other_Data')

# d2v 模型训练相关参数
default_d2v_trainConfig = edict()
default_d2v_trainConfig.Doc2Vec_File = 'kuaixue_cate_context.d2v'
default_d2v_trainConfig.Min_Count = 1
default_d2v_trainConfig.Window_Size = 3
default_d2v_trainConfig.Size = 2000
default_d2v_trainConfig.Workers = 10
default_d2v_trainConfig.Epoch_Num = 5

# 求相似度相关参数
default_simConfig = edict()
default_simConfig.Topn = 5
default_simConfig.WmdRate = 10
default_simConfig.simsmodel_name = 'tfidf'

# 特征工程相关参数
default_filterConfig = edict()
default_filterConfig.Length_Dif_Rate = 1
default_filterConfig.Same_Rate = 0.5

# 文件相关参数
filename_Config = edict()
filename_Config.context_file_name = 'kuaixue_cate_context.csv'
filename_Config.stoplist_name = 'stop_words.txt'
filename_Config.word_embedding = 'news_12g_baidubaike_20g_novel_90g_embedding_64.bin'
