# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Kubeflow Pipelines CUstomerChurn example

Run this script to compile pipeline
"""

import kfp.dsl as dsl
import kfp.gcp as gcp
import json

from kfp import components
from kfp.components import func_to_container_op

platform = 'GCP'


def convert_hp_results(experiment_result) -> str:
    import json
    r = json.loads(experiment_result)
    args = []
    for hp in r:
        print(hp)
        args.append("%s=%s" % (hp["name"], hp["value"]))

    return " ".join(args)


@dsl.pipeline(
    name='CustomerChurn',
    description='Customer Churn with DNN '
)
def cc_churn_hp_pipeline(gs_bucket='gs://your-bucket/export',
                         epochs=10,
                         batch_size=128,
                         input_data_file='<dir>/file.csv',
                         output_data_dir='output_dir',
                         model_dir='gs://your-bucket/export',
                         model_name='dummy',
                         server_name='dummy'):
    objectiveConfig = {
        "type": "maximize",
        "goal": 0.95,
        "objectiveMetricName": "accuracy"
    }
    algorithmConfig = {"algorithmName": "random"}
    parameters = [
        {"name": "--epochs", "parameterType": "int", "feasibleSpace": {"min": "4", "max": "20"}},
        {"name": "--batch_size", "parameterType": "int", "feasibleSpace": {"min": "20", "max": "200"}}
    ]
    rawTemplate = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": "{{.Trial}}",
            "namespace": "{{.NameSpace}}"
        },
        "spec": {
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {"name": "{{.Trial}}",
                         "image": "gcr.io/kube-2020/customerchurn/train:latest",
                         "command": [
                             "python /train.py --katib=1 {{- with .HyperParameters}} {{- range .}} {{.Name}}={{"
                             ".Value}} {{- end}} {{- end}} "
                         ]
                         }
                    ]
                }
            }
        }
    }
    trialTemplate = {
        "goTemplate": {
            "rawTemplate": json.dumps(rawTemplate)
        }
    }

    metricsCollectorSpec = {
        "collector": {
            "kind": "StdOut"
        }
    }

    katib_experiment_launcher_op = components.load_component_from_url(
        'https://raw.githubusercontent.com/kubeflow/pipelines/master/components/kubeflow/katib-launcher/component.yaml')

    hptune = katib_experiment_launcher_op(
        experiment_name=model_name,
        experiment_namespace="kubeflow",
        parallel_trial_count=3,
        max_trial_count=12,
        objective=str(objectiveConfig),
        algorithm=str(algorithmConfig),
        trial_template=str(trialTemplate),
        parameters=str(parameters),
        metrics_collector=str(metricsCollectorSpec),
        delete_finished_experiment=False)

    convert_op = func_to_container_op(convert_hp_results)
    op2 = convert_op(hptune.output)
    print(op2.output)

    preprocess_args = [
        '--bucket_name', gs_bucket,
        '--input_file', input_data_file,
        '--output_folder', output_data_dir
    ]
    preprocess = dsl.ContainerOp(
        name='preprocess',
        image='gcr.io/kube-2020/customerchurn/preprocess:latest',
        arguments=preprocess_args
    )
    train_args = [
        '--bucket_name', gs_bucket,
        '--epochs', epochs,
        '--batch_size', batch_size
    ]
    train = dsl.ContainerOp(
        name='train',
        image='gcr.io/kube-2020/customerchurn/train:latest',
        arguments=train_args
    )
    serve_args = [
        '--model_path', model_dir,
        '--model_name', model_name,
        '--server_name', server_name
    ]
    serve = dsl.ContainerOp(
        name='serve',
        image='gcr.io/kube-2020/customerchurn/serve:latest',
        arguments=serve_args
    )
    steps = [preprocess, train, serve, hptune]
    for step in steps:
        step.apply(gcp.use_gcp_secret('user-gcp-sa'))

        
    hptune.after(preprocess)
    op2.after(hptune)
    train.after(op2)
    serve.after(train)


if __name__ == '__main__':
    import kfp.compiler as compiler

    compiler.Compiler().compile(cc_churn_hp_pipeline, __file__ + '.tar.gz')
