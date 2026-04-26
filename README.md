# DSC232R Group Project: Data Exploration

## Dataset

**Dataset link:** [Smart Bee Colony Monitor: Clips of Beehive Sounds](https://www.kaggle.com/datasets/annajyang/beehive-sounds)

Our dataset is the **Smart Bee Colony Monitor: Clips of Beehive Sounds** dataset from Kaggle by Anna Yang. It consists of beehive audio recordings and related hive data collected from European honey bee hives in California under varying environmental and biological conditions, such as hive temperature and humidity.

The dataset is **23.21 GB**.

## Project Goal

Our objective is to develop a hive health monitoring system that classifies the overall condition of bee colonies based on audio patterns.

This project would be difficult to complete on a laptop because the dataset is very large and contains thousands of audio files that must be processed and analyzed. Distributed processing is needed because it allows the workload to be split across multiple computers, making the analysis faster and easier to manage.

## Project Members

- **Adham Kamel** — adkamel@ucsd.edu
- **Snigdha Tiwari** — sntiwari@ucsd.edu
- **Patcharapol Puckdee** — ppuckdee@ucsd.edu
- **Conner Houghtby** — choughtby@ucsd.edu

## SDSC Expanse Setup

We used SDSC Expanse with JupyterLab for this project.

- Account: TG-SEE260003
- Partition: shared
- Time limit: 120 minutes
- Cores: 8
- Memory per node: 128GB
- GPUs: 0
- Singularity image: `~/esolares/singularity_images/spark_py_latest_jupyter_dsc232r.sif`
- Module: `singularitypro`

## SDSC Expanse Setup

We used SDSC Expanse JupyterLab with 8 cores and 128GB memory per node.

### SparkSession Configuration

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "18g") \
    .config("spark.executor.instances", 7) \
    .config("spark.executor.cores", 1) \
    .getOrCreate()
