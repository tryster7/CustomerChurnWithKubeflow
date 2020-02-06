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

platform = 'GCP'


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
                         server_name='dummy',
                         goal=.92):
    objectiveConfig = {
        "type": "maximize",
        "goal": goal,
        "objectiveMetricName": "accuracy",
        "additionalMetricNames": ["test-loss"]
    }
    algorithmConfig = {"algorithmName": "random"}
    parameters = [
        {"name": "--epochs", "parameterType": "int", "feasibleSpace": {"min": "4", "max": "20"}},
        {"name": "--batch_size", "parameterType": "int", "feasibleSpace": {"min": "20", "max": "200"}},
        # {"name": "--optimizer", "parameterType": "categorical", "feasibleSpace": {"list": ["sgd", "adam", "ftrl"]}}
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

    hypertune = katib_experiment_launcher_op(
        model_name,
        "kubeflow",
        parallelTrialCount=3,
        maxTrialCount=12,
        objectiveConfig=str(objectiveConfig),
        algorithmConfig=str(algorithmConfig),
        trialTemplate=str(trialTemplate),
        parameters=str(parameters)
    )

    op_out = dsl.ContainerOp(
        name="hp-tuning-output",
        image="library/bash:4.4.23",
        command=["sh", "-c"],
        arguments=["echo hyperparameter: %s" % hypertune.output],
    )

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
    steps = [preprocess, train, serve, hypertune, op_out]
    for step in steps:
        step.apply(gcp.use_gcp_secret('user-gcp-sa'))

    hypertune.after(preprocess)
    op_out.after(hypertune)
    train.after(op_out)
    serve.after(train)


def katib_experiment_launcher_op(
        name,
        namespace,
        maxTrialCount=100,
        parallelTrialCount=3,
        maxFailedTrialCount=3,
        objectiveConfig='{}',
        algorithmConfig='{}',
        metricsCollector='{}',
        trialTemplate='{}',
        parameters='[]',
        experimentTimeoutMinutes=60,
        deleteAfterDone=False,
        outputFile='/hp_output.txt'):
    return dsl.ContainerOp(
        name="ccchurn-hpo",
        image='gcr.io/kube-2020/customerchurn/train:latest',
        arguments=[
            '--name', name,
            '--namespace', namespace,
            '--maxTrialCount', maxTrialCount,
            '--maxFailedTrialCount', maxFailedTrialCount,
            '--parallelTrialCount', parallelTrialCount,
            '--objectiveConfig', objectiveConfig,
            '--algorithmConfig', algorithmConfig,
            '--metricsCollector', metricsCollector,
            '--trialTemplate', trialTemplate,
            '--parameters', parameters,
            '--outputFile', outputFile,
            '--deleteAfterDone', deleteAfterDone,
            '--experimentTimeoutMinutes', experimentTimeoutMinutes,
        ],
        file_outputs={'bestHyperParameter': outputFile}
    )


if __name__ == '__main__':
    import kfp.compiler as compiler

    compiler.Compiler().compile(cc_churn_hp_pipeline, __file__ + '.tar.gz')
