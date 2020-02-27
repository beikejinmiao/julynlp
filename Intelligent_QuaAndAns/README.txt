1.allconfig.py：配置文件，设置全局变量的默认值，包括路径相关参数、模型相关参数、相似度相关参数、特征工程相关参数、文件相关参数

2.Filter_sentence.py：该文件用来对得到的句子进行筛选：
方法功能：
load_target_sentence：
    :param results: 模型返回的结果
    :param n: 用户需要匹配问题的个数
    :return: 得到我们要对其进行过滤的目标句子

get_same_rate：
    :param input_sentence: 用户输入的问题
    :param target_sentence: 要过滤的问题
    :return: 检测过滤问题和用户输入问题的字相同率

length_difference_rate：
    计算用户输入句子和匹配句子长度差异
    :param input_sentence:用户输入句子
    :param target_sentence:匹配句子
    :return:差异率

3.ES_model.py：ES模型
        :param inputfile: 用户上传文件
        :param index_name: 索引名
        :param doc_type_name: 文档名，默认为"context"
        :param stop_words_file: 停用词文件名
        :param index: bool，判断是否要重新建表
        :param model_name: 模型名称，默认为"es"
方法功能：
set_mapping：建立映射，即建立表结构，结构可以参见代码，在建立前，要先把原来的索引删除。
set_date：将用户文件中的问题插入表中，每条问题是json格式，通过'_index'指定索引，'_type'指定类型，'_source'指定问题的内容，有三部分：问题id、问题、分词结果
split_word：分词
build_model：创建模型，通过index参数指定是否需要重新建索引建表
get_topn_sims：得到最相似的n条问题，es要求输入问题的格式为：query = {'query': {"match": {"context": split_sent}}} es搜索过程主要代码在该方法体内有注释

4.Tfidf_model.py：TFIDF模型 与ES模型类似

5.Data.py：
get_stoplist：读取停用词
split_word：分词
split_dataset：对数据集所有问题进行切词，title列表示未切割问题，split_title表示已经切割
get_dataset：读取json格式的数据，如果路径中存在了json数据就直接读取，若没有就对原数据生成一个json格式数据，json格式数据为主要的数据输入输出格式
merge_json：将新增的数据加到总的json文件中

6.Intelligent_QuaAndAns:智能问答系统类
split_idAndcontext：        
        将 索引和文本内容分成两个csv文件存放
        :param context_file:
        :return:只有文本内容的csv文件名
bulid_model：
        构建模型
        :return:对应的模型对象

get_topn_sims：
        匹配相似度最高的问题
        :return:输出问题的id和匹配的问题 

该文件中类外方法：
parse_args：
基于用户id，创建关于该用户的csv、DataFrame、json、otherdata文件夹，将各种参数加入到parser对象中，可以统一管理修改各个参数，并且在Intelligent_QuaAndAns类中只需要
这个对象就能调用各个参数，然后将对象转换成命名空间对象，形如Namespace(参数=参数值,...)
return：返回该命名空间对象，形如Namespace(参数=参数值,...)

ThreadRun：本地智能问答系统类线程
   """
    :param is_build: 是否重新训练模型
    :param context_file_name: 上传文件名称
    :param stoplist_name: 停用词文件名
    :param input_sentence: 用户输入问题
    :param simsmodel_name: 使用模型
    :param topn: 返回多个问题
    :return: 问题的id和匹配的问题
    """
    当需要重新训练模型就基于用户数据重新训练，如果is_build==False就直接使用现有模型匹配问题

7.socker_server：服务接口
接收传入的请求，如果用户有输入问题，则使用模型匹配相似问题的id和内容，否则重新训练模型


