1.allconfig.py�������ļ�������ȫ�ֱ�����Ĭ��ֵ������·����ز�����ģ����ز��������ƶ���ز���������������ز������ļ���ز���

2.Filter_sentence.py�����ļ������Եõ��ľ��ӽ���ɸѡ��
�������ܣ�
load_target_sentence��
    :param results: ģ�ͷ��صĽ��
    :param n: �û���Ҫƥ������ĸ���
    :return: �õ�����Ҫ������й��˵�Ŀ�����

get_same_rate��
    :param input_sentence: �û����������
    :param target_sentence: Ҫ���˵�����
    :return: ������������û��������������ͬ��

length_difference_rate��
    �����û�������Ӻ�ƥ����ӳ��Ȳ���
    :param input_sentence:�û��������
    :param target_sentence:ƥ�����
    :return:������

3.ES_model.py��ESģ��
        :param inputfile: �û��ϴ��ļ�
        :param index_name: ������
        :param doc_type_name: �ĵ�����Ĭ��Ϊ"context"
        :param stop_words_file: ͣ�ô��ļ���
        :param index: bool���ж��Ƿ�Ҫ���½���
        :param model_name: ģ�����ƣ�Ĭ��Ϊ"es"
�������ܣ�
set_mapping������ӳ�䣬��������ṹ���ṹ���Բμ����룬�ڽ���ǰ��Ҫ�Ȱ�ԭ��������ɾ����
set_date�����û��ļ��е����������У�ÿ��������json��ʽ��ͨ��'_index'ָ��������'_type'ָ�����ͣ�'_source'ָ����������ݣ��������֣�����id�����⡢�ִʽ��
split_word���ִ�
build_model������ģ�ͣ�ͨ��index����ָ���Ƿ���Ҫ���½���������
get_topn_sims���õ������Ƶ�n�����⣬esҪ����������ĸ�ʽΪ��query = {'query': {"match": {"context": split_sent}}} es����������Ҫ�����ڸ÷���������ע��

4.Tfidf_model.py��TFIDFģ�� ��ESģ������

5.Data.py��
get_stoplist����ȡͣ�ô�
split_word���ִ�
split_dataset�������ݼ�������������дʣ�title�б�ʾδ�и����⣬split_title��ʾ�Ѿ��и�
get_dataset����ȡjson��ʽ�����ݣ����·���д�����json���ݾ�ֱ�Ӷ�ȡ����û�оͶ�ԭ��������һ��json��ʽ���ݣ�json��ʽ����Ϊ��Ҫ���������������ʽ
merge_json�������������ݼӵ��ܵ�json�ļ���

6.Intelligent_QuaAndAns:�����ʴ�ϵͳ��
split_idAndcontext��        
        �� �������ı����ݷֳ�����csv�ļ����
        :param context_file:
        :return:ֻ���ı����ݵ�csv�ļ���
bulid_model��
        ����ģ��
        :return:��Ӧ��ģ�Ͷ���

get_topn_sims��
        ƥ�����ƶ���ߵ�����
        :return:��������id��ƥ������� 

���ļ������ⷽ����
parse_args��
�����û�id���������ڸ��û���csv��DataFrame��json��otherdata�ļ��У������ֲ������뵽parser�����У�����ͳһ�����޸ĸ���������������Intelligent_QuaAndAns����ֻ��Ҫ
���������ܵ��ø���������Ȼ�󽫶���ת���������ռ��������Namespace(����=����ֵ,...)
return�����ظ������ռ��������Namespace(����=����ֵ,...)

ThreadRun�����������ʴ�ϵͳ���߳�
   """
    :param is_build: �Ƿ�����ѵ��ģ��
    :param context_file_name: �ϴ��ļ�����
    :param stoplist_name: ͣ�ô��ļ���
    :param input_sentence: �û���������
    :param simsmodel_name: ʹ��ģ��
    :param topn: ���ض������
    :return: �����id��ƥ�������
    """
    ����Ҫ����ѵ��ģ�;ͻ����û���������ѵ�������is_build==False��ֱ��ʹ������ģ��ƥ������

7.socker_server������ӿ�
���մ������������û����������⣬��ʹ��ģ��ƥ�����������id�����ݣ���������ѵ��ģ��


