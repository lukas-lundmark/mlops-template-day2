#!/usr/bin/env python3
import os
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Optional

from azureml.core import Environment, Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException


@dataclass(frozen=True)
class EnvironmentVariables:
    # Load the system environment variables
    load_dotenv()
    model_name: Optional[str] = os.environ.get("MODEL_NAME", "diamond-regressor")
    environment_name: Optional[str] = os.environ.get(
        "ENVIRONMENT_NAME", "conda-environment"
    )
    environment_file: Optional[str] = os.environ.get(
        "ENVIRONMENT_FILE", "environment_setup/ci_dependencies.yml"
    )
    scoring_dir: Optional[str] = os.environ.get("SCORING_DIR", "src")
    scoring_file: Optional[str] = os.environ.get("SCORING_FILE", "service/score.py")
    cpu_cluster: Optional[str] = os.environ.get("CPU_CLUSTER", 'my-cpu-cluster')
    service_name: Optional[str] = os.environ.get("SERVICE_NAME", 'my-diamond-regressor')


def get_environment(ws: Workspace, env_vars: EnvironmentVariables) -> Environment:
    environment_name = env_vars.environment_name
    assert environment_name is not None
    try:
        env = Environment.get(ws, name=environment_name)
    except Exception:
        assert env_vars.environment_file is not None
        env = Environment.from_conda_specification(
            name=environment_name, file_path=env_vars.environment_file
        )
        env.register(ws)
    return env

def get_cpu_cluster(ws: Workspace, env_vars: EnvironmentVariables) -> AmlCompute:
    cpu_cluster_name = env_vars.cpu_cluster
    try:
        aci_target = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    except ComputeTargetException:
        compute_config = AmlCompute.provisioning_configuration(
            vm_size="STANDARD_D1",
            max_nodes=1,
            idle_seconds_before_scaledown=300,
            tags={"Owner": "firstname.lastname@solita.fi", "DueDate": "01-07-2022"},
        )
        aci_target = ComputeTarget.create(ws, cpu_cluster_name, compute_config)
        aci_target.wait_for_completion(show_output = True)
    return aci_target
