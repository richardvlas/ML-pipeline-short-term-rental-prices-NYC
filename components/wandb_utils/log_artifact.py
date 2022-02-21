import wandb


def log_artifact(artifact_name, artifact_type, artifact_description, 
                 filename, wandb_run):
    """
    Log the provided filename as an artifact in W&B
    
    Parameters
    ----------
    artifact_name: str
        name for the artifact
    
    artifact_type: str
        type for the artifact (just a string like "raw_data", "clean_data" and 
        so on)
    
    artifact_description: str
        a brief description of the artifact
    
    filename: str
        local filename for the artifact

    wandb_run: Run object
        current Weights & Biases run
    
    Returns
    -------
    None
    """
    # Log to W&B
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        description=artifact_description,                
    )
    artifact.add_file(filename)
    wandb_run.log_artifact(artifact)
    # We need to call this .wait() method before we can use the
    # version below. This will wait until the artifact is loaded into W&B and a
    # version is assigned
    artifact.wait()
