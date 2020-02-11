TF Customer Churn prediction model running on Kubeflow and K8s

Handy kubectl cmds for inspecting distributed TF training jobs


1.	kubectl get tfjob.kubeflow.org/ccn-tf-dist -n kubeflow

2.	kubectl describe tfjob.kubeflow.org/ccn-tf-dist -n kubeflow

3.	kubectl get pods --selector=job-name=crn-tf-12 -n kubeflow

4.	kubectl get pods --selector=job-name=a-1tfjob -n kubeflow --output=jsonpath='{.items[*].metadata.name}'

5.	kubectl describe pod crn-tf-12-chief-0 -n kubeflow

6.	kubectl logs crn-tf-12-worker-1 -n kubeflow

7.	kubectl delete tfjob.kubeflow.org/ach-2-tf -n kubeflow
