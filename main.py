"""
Machine learning pipeline that runs all or just few selected parts of it 
and uploads the logs and artifacts to W&B platform
"""

import json
import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly:
    # "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(config_name="config")
def run_pipeline(config: DictConfig):
    """
    Run the pipeline or parts of it and uploads logs and artifacts
    to W&B platform

    Parameters
    ----------
    config : DictConfig
        configuration file

    Returns
    -------
    None
    """
    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config["main"]["steps"]
    active_steps = steps_par.split(',') if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        
        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                uri=f"{config['main']['component_repository']}/get_data",
                entry_point="main",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )
        
        if "basic_cleaning" in active_steps:
            _ = mlflow.run(
                uri=os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
                entry_point="main",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_sample",
                    "output_description": "Data with outliers and null values removed",
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

        if "data_check" in active_steps:
            _ = mlflow.run(
                uri=os.path.join(hydra.utils.get_original_cwd(), "src", "data_check"),
                entry_point="main",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

        







if __name__ == "__main__":
    run_pipeline()

