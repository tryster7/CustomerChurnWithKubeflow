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
import kfp.onprem as onprem

platform = 'GCP'


@dsl.pipeline(
    name='CustomerChurn',
    description='Customer Churn with DNN '
)
def customer_churn_pipeline(gs_bucket='gs://your-bucket/export',
                            epochs=10,
                            batch_size=128,
                            input_data_file='gs://your-bucket/export',
                            output_data_dir='gs://your-bucket/output',
                            model_dir='gs://your-bucket/export',
                            model_name='dummy',
                            server_name='dummy'):
    preprocess_args = [
        '--bucket_name', gs_bucket,
        '--input-file', input_data_file,
        '--output_folder', output_data_dir,
        '--batch_size', batch_size
    ]
    preprocess = dsl.ContainerOp(
        name='train',
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
    steps = [preprocess, train, serve]
    for step in steps:
        step.apply(gcp.use_gcp_secret('user-gcp-sa'))

    train.after(preprocess)
    serve.after(train)


if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(customer_churn_pipeline, __file__ + '.tar.gz')
