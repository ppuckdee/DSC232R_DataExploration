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

### First model: Decision Tree Classifier
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


### Links
[Preprocessing Notebook](https://github.com/ppuckdee/DSC232R-Beehive-Sound-Health-Monitoring/blob/e6241450506ab78798067f76d5b6e392b261070b/preprocess_milestone_three.ipynb)

## Milestone 4

### Second Model: PCA + Logistic Regression

For our second model, we used **PCA** for dimensionality reduction followed by **Logistic Regression** for classification.

PCA was used to shrink the feature set while keeping as much important information as possible. This is helpful because our dataset has many audio, hive, time, and environmental features. After reducing the features with PCA, we trained Logistic Regression to predict bee colony health.

### Evaluation

We evaluated the model using **training accuracy**, **test accuracy**, **F1 score**, and **PCA explained variance**.

The PCA notebook also compares training and test performance across different numbers of PCA components.

#### PCA Explained Variance

This graph shows how much information PCA keeps as more components are added.

<img width="889" height="490" alt="PCA explained variance plot" src="https://github.com/user-attachments/assets/5b63a35a-0f0d-43a4-95ac-a65629215365" />

### Fitting Analysis

This model is simpler than the Decision Tree from Milestone 3 because PCA reduces the number of features before Logistic Regression. This means it is less likely to overfit, but it could underfit if too much useful information was removed.

#### Training vs. Test Error

<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/c52720b8-d80a-4e50-8a39-f68ce3cb73f4" />

### Conclusion

The PCA + Logistic Regression model shows that dimensionality reduction can make the dataset smaller while still allowing us to classify bee colony health.

To improve this model, we could test different numbers of PCA components and compare Logistic Regression with other models.

### Speedup Analysis

We measured the performance of our feature engineering pipeline (data loading → preprocessing → aggregation) across different executor configurations.


### Methodology
- Dataset: 23.21GB Parquet files
- Operation: Full preprocessing pipeline
- Each measurement: Average of 3 runs

### Results

| Executors | Memory/Exec | Time (sec) | Speedup | Efficiency |
|-----------|-------------|------------|---------|------------|
| 1         | 64GB        | 0.8        | 1.00x   | 100%       |
| 3         | 20GB        | 0.77       | 1.03x   | 34%        |
| 7         | 14GB        | 0.78       | 1.03x   | 15%        |

#### Strong Scaling Analysis

This graph shows that adding more executors did not improve speed much.

<img width="1289" height="495" alt="image" src="https://github.com/user-attachments/assets/3038a522-c872-4317-a846-5e178693c632" />

### Analysis

Using the formula p = n(S-1) / S(n-1):
- With 7 executors: p = 7(1.03-1) / 1.03(6) = 0.034 (3.4% parallelizable)

Efficiency drops to 15% at 7 executors, indicating:
1. Shuffle overhead becomes significant
2. Memory per executor (14GB) may be insufficient
3. Amdahl's Law limits with ~5% sequential code

**Recommendation**: 2 executors will likely provide best balance of speedup and efficiency for our workload since the dropoff in efficiency between 1 and 3 executors is 66%.

### Links

[PCA Notebook](PCA.ipynb)


### Extra Credit
