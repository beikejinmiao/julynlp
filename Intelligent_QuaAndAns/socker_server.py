"""服务接口
"""
import json
import os
import socketserver

from Intelligent_QuaAndAns.main import ThreadRun


class Server(socketserver.BaseRequestHandler):

    def handle(self):
        connection = self.request
        print("connect request with %s..." % str(self.client_address))

        receive_data_json = connection.recv(1024).decode('utf-8')
        receive_data = json.loads(receive_data_json)
        is_build = receive_data['is_build']
        context_file_name = receive_data['context_file_name']
        simsmodel_name = receive_data['simsmodel_name']
        
        send_data = {}
        if 'input_sentence' in receive_data:
            # 匹配答案
            input_sentence = receive_data['input_sentence']
            
            try:
                question_id, results = ThreadRun(
                    is_build=is_build, context_file_name=context_file_name, input_sentence=input_sentence, simsmodel_name=simsmodel_name)

                send_data["statue"] = "success"
                send_data["question_id"] = int(question_id)
                send_data["results"] = results
            except Exception as e:
                send_data["statue"] = "error"
                send_data['message'] = str(e)
        else:
            # 模型训练
            try:
                ThreadRun(is_build=is_build, context_file_name=context_file_name,
                      simsmodel_name=simsmodel_name)
                send_data["statue"] = "success"
            except Exception as e:
                send_data["statue"] = "error"
                send_data['message'] = str(e)
        connection.send(json.dumps(send_data).encode('utf-8'))
        connection.close()


if __name__ == '__main__':
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 823), Server)
    print("Running socketserver...")
    server.serve_forever()
