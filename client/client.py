import sys
import json
import requests
import pandas as pd
from google.cloud import storage


def get_prediction(server_host='127.0.0.1', server_port=8500, model_name='ccd'):
    testX = pd.read_csv("gs://kube-1122/customerchurn/output/test.csv")
    testX = testX.drop(testX.columns[0], axis=1)
    print(testX.head())
    testy = pd.read_csv("gs://kube-1122/customerchurn/output/test_label.csv")
    testX = testy.drop(testy.columns[0], axis=1)
    print("predictions ...\n" , testy.head())
    data = json.dumps({"signature_name": "serving_default", "instances": testX[0:5].values.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://' + server_host + ':' + str(server_port) + '/v1/models/' + str(model_name) + ':predict',
                                  data=data, headers=headers)
    print(json_response)
    print(json.loads(json_response.text))


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: client server_host server_port ")
        sys.exit(-1)
    server_host = sys.argv[1]
    server_port = int(sys.argv[2])
    model_name = sys.argv[3]
    get_prediction(server_host=server_host, server_port=server_port)
