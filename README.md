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

### 1.1 Setup
We logged into SDSC Expanse via [portal.expanse.sdsc.edu](https://portal.expanse.sdsc.edu) 
and launched a JupyterLab session with the following configuration:

| Field | Value |
|-------|-------|
| Account | `TG-SEE260003` |
| Partition | `shared` |
| Number of Cores | 16 |
| Memory (GB) | 32 |
| Singularity Image | `~/esolares/singularity_images/spark_py_latest_jupyter_dsc232r.sif` |
| Environment Modules | `singularitypro` |
| Type | JupyterLab |

### 1.2 SparkSession Configuration

```python
spark = SparkSession.builder \
    .master("local[15]") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.instances", 15) \
    .getOrCreate()
```

### 1.3 Memory & Executor Justification

We requested **16 cores** and **32GB memory** for our 23GB beehive audio dataset.

Applying the formula from the course guidelines:
- **Driver memory**: 2GB (fixed)
- **Executor instances**: 16 - 1 = 15
- **Executor memory**: (32GB - 2GB) / 15 = 2GB each

We chose 16 cores and 32GB because:
- The dataset is approx 23GB of `.wav` audio files
- Audio decoding and feature extraction with `librosa` is CPU-bound, which benefits from parallelism across 15 executors
- 2GB per executor allows for sufficient headroom for audio processing
  
### Spark UI Screenshot

Placeholder: Insert Spark UI screenshot here.


## Milestone 3

### Preprocessing Plan
1A. Since the column `gust_speed` has 994 nulls (approx 78% of the dataset), we will drop this column. There is not enough data to compute anything meaningful. Additionally, `weather temp`, `wind speed`, `lat`, `long` have nulls but only 4 each. Thus, this can be computed to get meaningful conclusions but will need to be computed with the mean. 

### First model: Decision Tree Classifier**
- 80/20 random train/test split (seed=42).
- Baseline `DecisionTreeClassifier` at `maxDepth=5`, using `weightCol="weight"`
  so the minority classes aren't ignored.
- Train vs. test reported as accuracy, weighted F1, and error rate, with a
  confusion matrix on the test set.
- Depth sweep over `maxDepth ∈ {1, 2, 3, 5, 7, 10, 15}` tracking how
  training and test error diverge — the bias-variance fitting curve.
- Final summary at the depth that minimizes test error.

### Conclusion

The model fits cleanly: at modest depth the train and test curves track
each other within ~3–4%, and as depth grows the training error collapses
to zero 

### Improvements

**Drop identity/time features** (device, hive number, hour/month/day_of_week) to force the tree onto acoustic + sensor
  signal instead of hive-identity memorization.
