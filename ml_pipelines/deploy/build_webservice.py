#!/usr/bin/env python3
from azureml.core import Model, Workspace
from azureml.core.model import InferenceConfig
from argparse import ArgumentParser
from azureml.core.webservice import AciWebservice, LocalWebservice
from ml_pipelines.utils import EnvironmentVariables, get_environment, get_cpu_cluster

workspace = Workspace.from_config()
env_vars = EnvironmentVariables()

parser = ArgumentParser("Build Scoring Image")
parser.add_argument("--local", action="store_true")
args = parser.parse_args()

environment = get_environment(workspace, env_vars)
inference_config = InferenceConfig(
    entry_script=env_vars.scoring_file,
    source_directory=env_vars.scoring_dir,
    environment=environment,
)

model = Model(workspace, name=env_vars.model_name)

if not args.local:
    deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1, auth_enabled=True)
    aks_target = get_cpu_cluster(workspace, env_vars)
    print(aks_target)
    kwargs = {"deployment_target": aks_target}
else:
    deployment_config = LocalWebservice.deploy_configuration(port=6789)
    kwargs = {}

service = Model.deploy(
    workspace=workspace,
    name=env_vars.service_name,
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config,
    overwrite=True,
    **kwargs,
)
service.wait_for_deployment(show_output=True)
print("Scoring URI", service.scoring_uri)
print("Key", service.get_keys()[0])
