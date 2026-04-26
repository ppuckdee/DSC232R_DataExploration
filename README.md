# DSC232R: Beehive Sound Data Exploration and Health Monitoring

## Dataset

**Dataset link:** [Smart Bee Colony Monitor: Clips of Beehive Sounds](https://www.kaggle.com/datasets/annajyang/beehive-sounds)

We are using the **Smart Bee Colony Monitor: Clips of Beehive Sounds** dataset from Kaggle. It contains beehive audio recordings and hive information such as temperature and humidity.

The dataset is **23.21 GB**.

## Project Goal

Our goal is to use beehive sound data to classify the health condition of bee colonies.

This dataset is too large to process easily on a laptop because it contains many audio files. We are using distributed processing so the work can be split across multiple executors and processed more efficiently.

## Project Members

- **Adham Kamel** — adkamel@ucsd.edu
- **Snigdha Tiwari** — sntiwari@ucsd.edu
- **Patcharapol Puckdee** — ppuckdee@ucsd.edu
- **Conner Houghtby** — choughtby@ucsd.edu

## SDSC Expanse Setup

We used SDSC Expanse JupyterLab with:

- 8 cores
- 128GB memory per node

### SparkSession Configuration

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "18g") \
    .config("spark.executor.instances", 7) \
    .config("spark.executor.cores", 1) \
    .getOrCreate()
```

### Spark UI Screenshot

Placeholder: Insert Spark UI screenshot here.
