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

### Cookie cutter
In order to make your job a little easier, you are provided a cookie cutter template that you can use to create 
stubs for new pipeline components. It is not required that you use this, but it might save you from a bit of 
boilerplate code. Just run the cookiecutter and enter the required information, and a new component 
will be created including the `conda.yml` file, the `MLproject` file as well as the script. You can then modify these
as needed, instead of starting from scratch.
For example:

```bash
cookiecutter cookie-mlflow-step -o src
```

```bash
step_name [step_name]: basic_cleaning
script_name [run.py]: run.py
job_type [my_step]: basic_cleaning
short_description [My step]: This steps cleans the data
long_description [An example of a step using MLflow and Weights & Biases]: Performs basic cleaning on the data and save the results in Weights & Biases
parameters [parameter1,parameter2]: parameter1,parameter2,parameter3
```

This will create a step called ``basic_cleaning`` under the directory ``src``. Type this command:

```bash
ls src/basic_cleaning/
```

which returns the following structure:

```bash
conda.yml  MLproject  run.py
```


You can now modify the script (``run.py``), the conda environment (``conda.yml``) and the project definition 
(``MLproject``) as you please.

The script ``run.py`` will receive the input parameters ``parameter1``, ``parameter2``,
``parameter3`` and it will be called like:

```bash
mlflow run src/step_name -P parameter1=1 -P parameter2=2 -P parameter3="test"
```
