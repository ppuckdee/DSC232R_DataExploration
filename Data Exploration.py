from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ── SparkSession (this is from the example in the project explanation - idk how we did it otherwise) ─
spark = SparkSession.builder \
    .appName("BeehiveExploration") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "18g") \
    .config("spark.executor.instances", 7) \
    .getOrCreate()


df = spark.read.csv("all_data_updated.csv", header=True, inferSchema=True)
       

CONTINOUS_COLS = [
    "hive temp", "hive humidity", "hive pressure",
    "weather temp", "weather humidity", "weather pressure",
    "wind speed", "gust speed", "cloud coverage",
    "rain", "lat", "long", "frames", "time"
]

CATEGORICAL_COLS = [
    "device", "hive number", "weatherID",
    "queen presence", "queen acceptance",
    "queen status", "target"
]


# 3A

total_rows = df.count()
total_columns = len(df.columns)

print(f"Total observations (rows): {total_rows:,}")
print(f"Total columns: {total_columns}")

# 3B

# Print schema
print("=" * 60)
print("SCHEMA")
print("=" * 60)
df.printSchema()

# Describe statistics
print("=" * 60)
print("Description Statistics")
print("=" * 60)
df.select([F.col(f"`{c}`") for c in CONTINOUS_COLS]).describe().show(truncate=False)

print("=" * 60)
print("Quantiles")
print("=" * 60)
quantile_aggs = []
for c in CONTINOUS_COLS:
    quantile_aggs += [
        F.expr(f"percentile_approx(`{c}`, 0.25)").alias(f"{c}_Q1"),
        F.expr(f"percentile_approx(`{c}`, 0.50)").alias(f"{c}_median"),
        F.expr(f"percentile_approx(`{c}`, 0.75)").alias(f"{c}_Q3"),
    ]
df.agg(*quantile_aggs).show(vertical=True, truncate=False)


print("=" * 60)
print("Categorical Column Distribution")
print("=" * 60)
for col in CATEGORICAL_COLS:
    print(f"\n--- {col} ---")
    df.groupBy(col) \
      .agg(
          F.count("*").alias("count"),
          F.round(F.count("*") / total_rows * 100, 2).alias("pct_%")
      ) \
      .orderBy(col) \
      .show(truncate=False)


print("=" * 60)
print("Unique Value Counts")
print("=" * 60)
for col in CATEGORICAL_COLS:
    n = df.select(F.col(f"`{col}`")).distinct().count()
    print(f"  {col}: {n} unique values")



# 3C

print("=" * 60)
print("Null Value Counts Per Column")
print("=" * 60)
for c in df.columns:
    null_count = df.filter(F.col(f"`{c}`").isNull()).count()
    null_pct = round(null_count / total_rows * 100, 2)
    print(f"{c:30s}  nulls: {null_count:4d}  ({null_pct}%)")


print("=" * 60)
print("Duplicate Row Check")
print("=" * 60)
distinct_rows = df.distinct().count()
duplicate_count = total_rows - distinct_rows

print(f"Total rows:     {total_rows:,}")
print(f"Distinct rows:  {distinct_rows:,}")
print(f"Duplicate rows: {duplicate_count:,}")





