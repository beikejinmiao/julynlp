from elasticsearch import Elasticsearch

# 默认host为localhost,port为9200.但也可以指定host与port  hosts={"host": '0.0.0.0', "port": 9200}
es = Elasticsearch()

# 添加或更新数据,index，doc_type名称可以自定义，id可以根据需求赋值,body为内容
# es.index(index="my_index",doc_type="test_type",id=1,body={"name":"python","addr":"深圳"})
#
# # 或者:ignore=409忽略文档已存在异常
# es.create(index="my_index",doc_type="test_type",id=1,ignore=409,body={"name":"python","addr":"深圳"})
#es.delete(index='kuaixue', doc_type='context', id=1)
es.indices.delete(index='kuaixue', ignore=[400, 404])
# es.indices.delete(index='my_index', ignore=[400, 404])
# doc = [
#      {"index": {}},
#      {'name': 'jackaaa', 'age': 2000, 'sex': 'female', 'address': u'北京'},
#     ]
# es.bulk(index='kuaixue', doc_type='context-splitword', body=doc)

#es.index(index="my_index",doc_type="test_type",id=1,body={"name":"python","addr":"深圳"})
# # 获取索引为my_index,文档类型为test_type的所有数据,result为一个字典类型
#result = es.search(index="my_index",doc_type="test_type")

# # 或者这样写:搜索id=1的文档
# result = es.get(index="my_index",doc_type="test_type",id=20)
#
# #打印所有数据
# print(result)
# for item in result["hits"]["hits"]:
#     print(item["_source"])  #{'name': 'python', 'addr': '深圳'}


# query = {'query': {'match': {'split_word':"税负"}}}# 查找年龄大于11的所有文档
# allDoc = es.search(index='my_index', doc_type='test_type', body=query)

# query = {'query':{"bool":{
#       "must": [
#         { "match": { "split_word": "老师" } },
#         { "match": { "split_word": "税负" } }
#       ]
#     }}}# 查找年龄大于11的所有文档
# allDoc = es.search(index='my_index', doc_type='test_type', body=query)

# query = {'query': { "constant_score" : {
#             "filter" : {
#                 "terms" : {
#                     "split_word" : ['预期']
#                 }
#             }
#         }}}

query = {'query': {"match" : {"context" : '老师 物流 税负 率'}}}
#
allDoc = es.search(index='kuaixue', doc_type='context', body=query)
print(allDoc)
for item in allDoc["hits"]["hits"]:
    print(item["_source"]["context"])