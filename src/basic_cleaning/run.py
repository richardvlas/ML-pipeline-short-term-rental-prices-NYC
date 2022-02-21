#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def basic_cleaning(args):
    """
    Data processing that starts from raw data artifact and creates a new 
    artifact clean_sample.csv with the cleaned data that is uploaded to W&B

    Parameters
    ----------
    args: Namespace
        Command line argument strings object assigned as attributes of 
        the namespace. 

    Returns
    -------
    None
    """

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    logger.info(f"Donwloading input artifact: {args.input_artifact}")
    input_artifact = run.use_artifact(args.input_artifact)
    input_artifact_path = input_artifact.file()

    # Load artifact into pandas DataFrame
    df = pd.read_csv(input_artifact_path)

    # Drop outliers in price column
    logger.info(f"Cleaning artifact - price column filter between min/max range ({args.min_price}, {args.max_price})")
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()

    # # Drop outliers in longitude/latitude column
    # logger.info("Cleaning artifact - longitude/latitude column filter between min/max range")
    # min_longitude = -74.25
    # max_longitude = -73.50
    # min_latitude = 40.5
    # max_latitude = 41.2
    # idx = (df['longitude'].between(min_longitude, max_longitude)) & \
    #     (df['latitude'].between(min_latitude, max_latitude))
    # df = df[idx].copy()

    # Convert last_review column to datetime
    logger.info("Cleaning artifact - converting last_review column to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Save the cleaned dataset to local csv file
    logger.info(f"Saving cleaned data to {args.output_artifact}")
    df.to_csv(args.output_artifact, index=False)

    # Logging artifact to W&B
    logger.info(f"Logging artifact {args.output_artifact}")
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(args.output_artifact)

    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name for the input artifact stored at W&B",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name for the W&B artifact that will be created",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type for the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description for the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum value for price",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum value for price",
        required=True
    )


    args = parser.parse_args()

    basic_cleaning(args)
