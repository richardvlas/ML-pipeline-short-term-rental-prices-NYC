# Build an ML Pipeline for Short-term Rental Prices in NYC

You are working for a property management company renting rooms and properties for short periods of time on various rental platforms. You need to estimate the typical price for a given property based on the price of similar properties. Your company receives new data in bulk every week. The model needs to be retrained with the same cadence, necessitating an end-to-end pipeline that can be reused.

In this project you will build such a pipeline. Here is a diagram of the ML pipeline containing all components and resulting artifacts that you are going to build:

## TODO: ADD DEVELOPMENT PIPELINE !!

### Links to W&B project and Github repository

```bash
TODO: Add link to W&B project
https://github.com/richardvlas/First-MLflow-Pipeline
```

## Project Structure
The following represents the structure of folders and files contained in the project after runing the pipeline.

> Note: A lot of files generated by wandb and Mlflow were not pushed to this repo to keep only the code of the ML pipeline. Only artifacts generated are kept.

```bash
TODO: Add project structure
```

## Preliminary steps
### Clone the Project
Clone the repository locally so that you can use it:

```bash
git clone https://github.com/richardvlas/ML-pipeline-short-term-rental-prices-NYC.git
```
and go into the repository:

```bash
cd ML-pipeline-short-term-rental-prices-NYC
```

### Create environment
Make sure to have conda installed and ready, then create a new environment using the `environment.yml` file provided in the root of the repository and activate it:

```bash
conda env create -f environment.yml
conda activate nyc_airbnb_dev 
```

### Get API key for Weights and Biases

Let's make sure you are logged in to Weights & Biases. Get your API key from W&B by going to https://wandb.ai/authorize and click on the + icon (copy to clipboard), then paste your key into this command:

```bash
$ wandb login [your API key]
```

You should see a message similar to:

```
wandb: Appending key for api.wandb.ai to your netrc file: /home/[your username]/.netrc
```
