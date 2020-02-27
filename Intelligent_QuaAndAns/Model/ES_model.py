import jieba
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


class ES_Model(object):
    def __init__(self, inputfile, index_name, doc_type_name, stop_words_file, index, model_name='es'):
        self.stop_words_file = stop_words_file
        with open(self.stop_words_file, encoding='utf-8') as f:
            lines = f.readlines()
        self.stop_list = [x.strip() for x in lines]

        self.index = index
        self.inputfile = inputfile
        self.model_name = model_name
        self.index_name = index_name
        self.doc_type_name = doc_type_name
        self.build_model()

    def set_mapping(self, es):
        my_mapping = {"mappings": {self.doc_type_name: {
            "properties": {"context": {"type": "text"}, "split_word": {"type": "text"}, "context_id": {"type": "text"}}}}}
        # 创建Index和mapping
        
        es.indices.delete(index=self.index_name, ignore=[400, 404])
        create_index = es.indices.create(
            index=self.index_name, body=my_mapping)  # {u'acknowledged': True}

        if not create_index["acknowledged"]:
            print("Index data bug...")

    # 将文件中的数据存储到es中
    def set_data(self, es, input_file):
        # 读入数据
        with open(input_file, 'r', encoding='utf-8') as line_list:
            # 创建ACTIONS
            ACTIONS = []
            for line in line_list:
                fields = line.split(',')
                print("fields:", fields)
                # print fields[1]
                action = {
                    "_index": self.index_name,
                    "_type": self.doc_type_name,
                    "_source": {
                        "context": fields[1],
                        "split_word": fields[2].rstrip(),
                        "context_id": fields[0]
                    },
                }
                ACTIONS.append(action)
                # 批量处理
            success, _ = bulk(
                es, ACTIONS, index=self.index_name, raise_on_error=True)
            print('Performed %d actions' % success)

    @staticmethod
    def split_word(query, stop_list):
        words = jieba.cut(query)
        result = ' '.join([word for word in words if word not in stop_list])
        return result

    def build_model(self):
        if self.index:
            print("build model.....")
            print("build Elasticsearch.....")
            print("Indexing data.....")
            self.es = Elasticsearch()
            self.set_mapping(self.es)
            self.set_data(self.es, self.inputfile)
            print("Index data success")
            print("build model success")
        else:
            print("build model.....")
            print("build Elasticsearch.....")
            self.es = Elasticsearch()
            print("build model success")

    def get_topn_sims(self, sentences, n=5):
        split_sent = self.split_word(sentences, self.stop_list)
        results_1 = {'title': sentences, 'split_title': split_sent}
        # print("split_sent:",split_sent)
        query = {'query': {"match": {"context": split_sent}}}

        allDoc = self.es.search(index=self.index_name,
                                doc_type=self.doc_type_name, body=query)
        print (allDoc)

        results_2 = {}
        items = allDoc["hits"]["hits"]
        # print("items:",items)

        for item in items:
            print(item["_source"]["split_word"])

        for i in range(n):
            each_results_2 = {'index': str(int(items[i]["_source"]["context_id"])+1), 'similarity': str(items[i]["_score"]),
                              'title': items[i]["_source"]["context"],
                              'confidence': None}
            results_2[i] = each_results_2
        results = {'result1': results_1, 'result2': results_2}
        return results
