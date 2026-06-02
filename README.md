# DSC232R: Beehive Sound Data Exploration and Health Monitoring

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Project Members](#project-members)
4. [Environment Setup](#environment-setup)
5. [Methods](#methods)
6. [Results](#results)
7. [Figures](#figures)
8. [Discussion](#discussion)
9. [Conclusion](#conclusion)
10. [Notebooks](#notebooks)

---

## Introduction

Bee colonies are responsible for pollinating roughly one-third of the global food supply. Monitoring hive health is difficult because beekeepers have to physically inspect each hive, which takes time and disturbs the colony. Research has shown that the sounds a hive makes — the buzzing frequency and intensity — can indicate stress events like queenlessness, disease, or swarming. A model that can automatically classify hive health from audio would allow beekeepers to monitor many hives at once without manual inspection.

This project requires big data tools for two reasons. First, the dataset is **23.21 GB** of `.wav` audio recordings, which is too large to process on a single machine without running out of memory. Second, extracting audio features (like MFCCs) from each file one at a time would take hours. We used Spark on SDSC Expanse to distribute the feature extraction across 15 executors, which made it possible to work with the full dataset and iterate on models in a reasonable amount of time. Without distributed computing, this project would not be practical.

---

## Dataset

**Dataset:** [Smart Bee Colony Monitor: Clips of Beehive Sounds](https://www.kaggle.com/datasets/annajyang/beehive-sounds)

We are using the **Smart Bee Colony Monitor: Clips of Beehive Sounds** dataset from Kaggle. It contains beehive audio recordings along with hive sensor readings (temperature, humidity, pressure) and weather data. Each audio clip is labeled with one of six health condition classes (0–5). After feature extraction, the dataset has approximately 1,275 labeled samples.

The dataset is **23.21 GB**.

---

## Project Members

- **Adham Kamel** — adkamel@ucsd.edu
- **Snigdha Tiwari** — sntiwari@ucsd.edu
- **Patcharapol Puckdee** — ppuckdee@ucsd.edu
- **Conner Houghtby** — choughtby@ucsd.edu

---

## Environment Setup

We logged into SDSC Expanse via [portal.expanse.sdsc.edu](https://portal.expanse.sdsc.edu) and launched a JupyterLab session with the following configuration:

| Field | Value |
|---|---|
| Account | `TG-SEE260003` |
| Partition | `shared` |
| Number of Cores | 16 |
| Memory (GB) | 32 |
| Singularity Image | `~/esolares/singularity_images/spark_py_latest_jupyter_dsc232r.sif` |
| Environment Modules | `singularitypro` |
| Type | JupyterLab |

### SparkSession Configuration

```python
spark = SparkSession.builder \
    .master("local[15]") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.instances", 15) \
    .getOrCreate()
```

### Memory & Executor Justification

We requested **16 cores** and **32GB memory** for our 23GB beehive audio dataset.

Applying the formula from the course guidelines:
- **Driver memory**: 2GB (fixed)
- **Executor instances**: 16 - 1 = 15
- **Executor memory**: (32GB - 2GB) / 15 = 2GB each

We chose 16 cores and 32GB because:
- The dataset is approx 23GB of `.wav` audio files
- Audio decoding and feature extraction with `librosa` is CPU-bound, which benefits from parallelism across 15 executors
- 2GB per executor allows for sufficient headroom for audio processing

---

## Methods

### Data Exploration

We started by loading the raw CSV metadata and looking at the schema and null counts across all columns. The main findings were:

- `gust_speed` had **994 nulls (~78% of rows)** so we dropped it entirely since there wasn't enough data to use it meaningfully.
- `weather_temp`, `wind_speed`, `lat`, and `long` each had **4 nulls**, which we filled in using the column mean.
- The health class labels were **imbalanced** — class 3 appeared much more often than the others, while classes 0 and 2 were minority classes.

### Preprocessing (using Spark)

We extracted audio features from each `.wav` file using a Spark UDF that wrapped `librosa`. This produced MFCCs, spectral centroid, and chroma features for each clip. We then joined those features with the sensor and weather data from the CSV. All features were assembled and scaled using a Spark pipeline:

```python
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler    = StandardScaler(inputCol="features", outputCol="features_scaled",
                           withMean=True, withStd=True)
pipeline  = Pipeline(stages=[assembler, scaler])
```

To handle class imbalance, we computed class weights from the label distribution and added them as a `weight` column. The processed features were saved to Parquet on Expanse's Lustre filesystem. We then split the data **80/20** (seed=42), giving us **1,053 training rows** and **222 test rows**.

### Model 1: Decision Tree Classifier

Our first model was a `DecisionTreeClassifier` from PySpark MLlib, trained with class weights so minority classes weren't ignored. We ran a depth sweep over `maxDepth ∈ {1, 2, 3, 5, 7, 10, 15}` to see how training and test error changed, and used that to produce a bias-variance fitting curve.

```python
dt = DecisionTreeClassifier(
    labelCol="target",
    featuresCol="features",
    maxDepth=5,
    weightCol="weight"
)
```

We reported accuracy, weighted F1, and error rate for both train and test at each depth, along with a confusion matrix on the test set.

### Model 2: PCA + Logistic Regression

For our second model, we used **PCA** (`pyspark.ml.feature.PCA`) to reduce the number of features, followed by **Logistic Regression** to classify hive health. We first fit PCA with k=20 to see how variance was distributed across components, then chose **k=10** as our operating point since it captured 73.85% of the variance. We also ran a sweep over k ∈ {2, 5, 10, 15, 20} to produce a fitting curve similar to the one we made for Model 1.

```python
pca = PCA(k=10, inputCol="features", outputCol="pca_features")
lr  = LogisticRegression(
    featuresCol="pca_features",
    labelCol="target",
    weightCol="weight",
    maxIter=100
)
pipeline = Pipeline(stages=[pca, lr])
```

The reason we tried this model after the Decision Tree is that PCA removes correlated and noisy features before classification, which helps prevent the model from memorizing the training data.

---

## Results

### Model 1: Decision Tree

The Decision Tree at `maxDepth=5` generalized well, with training and test error staying within **~3–4%** of each other at shallow depths. As we increased depth, training error dropped toward zero while test error leveled off — the classic sign of overfitting. We selected the depth that gave the lowest test error as our final model.

### Model 2: PCA + Logistic Regression

#### Explained Variance by PCA Components

This table shows how much variance each group of components captures.

| Components (k) | Variance Explained | Cumulative |
|---|---|---|
| 1 | 17.35% | 17.35% |
| 2 | 12.07% | 29.42% |
| 5 | 5.95% | 52.63% |
| 10 | 2.90% | 73.85% |
| 15 | — | 86.16% |
| 20 | — | ~93% |

#### Accuracy Metrics (k=10)

This table shows the training and test scores for the PCA + Logistic Regression model.

<img width="470" height="74" alt="model2metrics" src="https://github.com/user-attachments/assets/f3f7a8ef-6551-4fa7-8c53-d6da8bb1f0fb" />

|  | Train | Test | Gap |
|---|---|---|---|
| Accuracy | 0.7056 | 0.7117 | −0.006 |
| F1 (weighted) | 0.7122 | 0.7181 | −0.006 |
| Error Rate | 0.2944 | 0.2883 | −0.006 |

#### Fitting Curve: Error vs. Number of PCA Components

This table shows how training and test error change as we add more PCA components.

| k | Train Error | Test Error |
|---|---|---|
| 2 | 0.3599 | 0.4099 |
| 5 | 0.3485 | 0.3964 |
| 10 | 0.2944 | 0.2883 |
| 15 | 0.1985 | 0.2252 |
| 20 | 0.1311 | 0.1667 |

The best test error was at **k=20** (16.67% error), but we used k=10 as the main model since it captures most of the important variance with fewer components.

<img width="790" height="490" alt="PCA Logistic Regression training vs test error graph" src="https://github.com/user-attachments/assets/c52720b8-d80a-4e50-8a39-f68ce3cb73f4" />

#### Predictions: Correct Classifications, False Positives, and False Negatives

This confusion matrix shows how the model performed on the test set (222 samples). Rows are the true class, columns are the predicted class.

| True \ Pred | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| **0** | 11 | 1 | 1 | 1 | 3 | 4 |
| **1** | 0 | 9 | 4 | 2 | 0 | 0 |
| **2** | 0 | 4 | 10 | 0 | 0 | 0 |
| **3** | 5 | 6 | 3 | 57 | 0 | 0 |
| **4** | 1 | 0 | 0 | 1 | 20 | 12 |
| **5** | 6 | 0 | 0 | 1 | 9 | 51 |

- **Correct classifications:** Class 3 (57/71), Class 5 (51/67), and Class 4 (20/34) were predicted best, which makes sense since they are the larger classes.
- **False negatives:** Class 0 had the worst miss rate, and 10 out of 21 samples were classified as something else. This is a concern because it means the model would fail to flag unhealthy hives.
- **False positives:** Classes 4 and 5 were frequently confused with each other (12 class-4 samples predicted as 5, and 9 class-5 samples predicted as 4), which suggests these two health states have similar audio profiles.

#### Speedup Analysis

We measured how long the preprocessing pipeline took (data loading → aggregation) across three different executor configurations. Each result is the average of 3 runs with the first run discarded to account for JVM warmup.

| Executors | Memory/Exec | Time (s) | Speedup | Efficiency |
|---|---|---|---|---|
| 1 | 64 GB | 0.80 | 1.00× | 100% |
| 3 | 20 GB | 0.77 | 1.03× | 34% |
| 7 | 14 GB | 0.78 | 1.03× | 15% |

Using the formula p = n(S-1) / S(n-1): with 7 executors, p = 7(1.03-1) / 1.03(6) = 0.034, meaning only about 3.4% of the pipeline is parallelizable.

<img width="1289" height="495" alt="strong scaling analysis graph" src="https://github.com/user-attachments/assets/3038a522-c872-4317-a846-5e178693c632" />

---

## Figures

| Figure | Location | Description |
|---|---|---|
| PCA Explained Variance | [model2_pca_logistic_regression.ipynb](model2_pca_logistic_regression.ipynb) | Cumulative variance retained vs. number of components (k=1–20) |
| Model 2 Metrics Table | [model2_pca_logistic_regression.ipynb](model2_pca_logistic_regression.ipynb) | Training and test accuracy, F1, and error rate at k=10 |
| Fitting Curve (Model 2) | [model2_pca_logistic_regression.ipynb](model2_pca_logistic_regression.ipynb) | Train and test error across k ∈ {2, 5, 10, 15, 20} |
| Confusion Matrix (Model 2) | [model2_pca_logistic_regression.ipynb](model2_pca_logistic_regression.ipynb) | Test-set predictions for PCA + Logistic Regression at k=10 |
| Fitting Curve (Model 1) | [model1_decision_tree.ipynb](model1_decision_tree.ipynb) | Train and test error vs. Decision Tree maxDepth |
| Strong Scaling Plot | [model2_pca_logistic_regression.ipynb](model2_pca_logistic_regression.ipynb) | Measured vs. Amdahl-predicted speedup at 1, 3, 7 executors |

<img width="889" height="490" alt="PCA explained variance plot" src="https://github.com/user-attachments/assets/5b63a35a-0f0d-43a4-95ac-a65629215365" />

---

## Discussion

### Data Exploration & Preprocessing

Dropping `gust_speed` was straightforward since 78% of values were missing, so there wasn't enough data to get anything useful from it. The other four columns with nulls were small enough that filling them with the column mean shouldn't affect results much. The bigger challenge was class imbalance. Without class weighting, both models would have been biased toward predicting class 3 (the most common label) and would have ignored minority classes. The weighting helps balance this out, though it doesn't fully solve the problem of having fewer examples for some classes.

### Model 1: Decision Tree

The Decision Tree worked well at shallow depths, which makes sense because health classes likely have clear thresholds in sensor readings like temperature and humidity that tree splits can pick up on. The issue at higher depths is that the model starts memorizing things like which specific hive a recording came from (device ID, hive number) or what time it was recorded (hour, month), rather than learning from the audio signal itself. These features are specific to the hives in our training set and won't generalize to new hives. Dropping them would likely reduce overfitting and give a more honest accuracy estimate.

### Model 2: PCA + Logistic Regression

At k=10 the model is in the **well-fitted region** of the fitting graph, meaning the gap between training and test accuracy is almost zero (−0.006). This makes sense because the dataset is small (~1,275 samples) and Logistic Regression with regularization is unlikely to overfit on only 10 features.

Increasing k beyond 10 keeps reducing both train and test error, and k=20 gives the best test result at 16.67% error. This suggests more components would help, but we'd need a larger dataset to push further without risking overfitting.

The confusion matrix shows two clear weaknesses. First, **classes 4 and 5 are frequently confused with each other**, which suggests these health states have similar audio profiles that a linear model can't separate well. Second, **class 0 has the worst recall**, with only 11 out of 21 samples classified correctly. From a real-world perspective this is the most important issue, since missing an unhealthy hive defeats the purpose of the system. Both of these weaknesses point to the limits of using a linear decision boundary, and a nonlinear classifier would likely do better here.

Compared to Model 1, the PCA + Logistic Regression model generalizes more consistently, as the Decision Tree's train/test gap grew significantly at higher depths while the linear model's gap stayed near zero across all values of k we tested.

### Speedup Analysis

Adding more executors barely helped, since speedup was only 1.03× at both 3 and 7 executors. This is because the step we measured (CSV/Parquet loading and aggregation) is mostly sequential, with only about 3.4% of it parallelizable according to Amdahl's Law. The audio feature extraction UDF, which is the actual expensive step since it runs per file, would likely show much better scaling since it can be distributed across files independently. We didn't isolate that step in the speedup experiment, so the results here underrepresent how much Spark actually helps for this project.

---

## Conclusion

Our goal was to classify beehive health from audio and sensor data using distributed machine learning on SDSC Expanse. We built two models across the project:

**Model 1 — Decision Tree:** This model generalized well at shallow depths, with train and test error staying within ~3–4% of each other. It's easy to interpret and picked up on clear patterns in the sensor data, but it tends to overfit at higher depths and may have relied on hive-identity features rather than the audio signal itself.

**Model 2 — PCA + Logistic Regression:** This model achieved 71.2% test accuracy with almost no gap between training and test performance. PCA reduced the feature space to 10 components while keeping 73.85% of the variance, giving Logistic Regression a cleaner input to work with. The main weaknesses were the confusion between classes 4 and 5 and the poor recall for class 0, both of which are limitations of using a linear classifier.

**What we learned about big data processing:** Not every part of a pipeline benefits equally from more executors. Aggregation tasks showed almost no speedup, while per-file audio processing would scale much better. Understanding where the bottlenecks actually are matters as much as knowing how to use the framework.

**How distributed computing changed our approach:** Without Spark and Expanse, we would have had to work with a small sample of the audio files. Being able to process the full 23 GB dataset gave us reliable class distributions for weighting and made it practical to run model sweeps (depth sweep, component sweep) in a single session.

**What we would explore with more time:**
- Nonlinear classifiers like Random Forest or Gradient Boosting on PCA-reduced features to better handle the class 4/5 boundary
- Richer audio features like mel spectrograms or learned embeddings from a pretrained audio model
- Increasing k beyond 20 with stronger regularization to push test error lower
- Oversampling minority classes or applying a higher penalty on class 0 false negatives
- Measuring speedup specifically on the audio UDF step, where parallelism is expected to make a real difference

---

## Notebooks

| Notebook | Description |
|---|---|
| [Data Exploration](data_exploration.ipynb) | Schema inspection, descriptive statistics, null analysis, class distribution |
| [Preprocessing + Decision Tree (Model 1)](model1_decision_tree.ipynb) | Audio feature extraction, preprocessing pipeline, Model 1 training and evaluation |
| [PCA + Logistic Regression (Model 2)](model2_pca_logistic_regression.ipynb) | Dimensionality reduction, model training, evaluation, fitting curve, speedup analysis |
| [Extra Credit: Spark vs Ray](extra_credit.ipynb) | Framework comparison on statistical aggregation tasks |

### Extra Credit: Spark vs. Ray Framework Comparison

**Which framework was faster?**
Spark was faster for this task. The beehive CSV sits on Expanse's Lustre filesystem, and `describe()` + `percentile_approx` are single-pass aggregations that Spark handles efficiently in one call. Ray reads the CSV sequentially before distributing the work, so the overhead ends up outweighing any parallelism benefit at this dataset size. Spark averaged 1.01s vs Ray's 142.94s.

**Which was easier to implement?**
Spark was easier to implement. `df.describe()` and `percentile_approx` produce a clean summary in one call. Ray Data doesn't have a built-in percentile aggregator, so we had to convert to pandas to compute quantiles, which added an extra step and made the code more complicated.

**Which would we choose for this use case?**
We would choose Spark for tabular statistics and preprocessing. Ray would be the better choice if the pipeline included the audio feature extraction UDF at scale or model training with PyTorch, since Ray's integrations (Ray Train, Ray Tune) give it a clear advantage over Spark's MLlib for those kinds of tasks.

---

## Statement of Collaboration

- Snigdha Tiwari: Contributor: Did data exploration, preprocessing, final repo cleanup, extra credit
- Adham Kamel: Title: Contribution
- Patcharapol Puckdee: Title: Contribution
- Conner Houghtby: Title: Contribution
