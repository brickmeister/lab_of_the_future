# Databricks notebook source
# MAGIC %md
# MAGIC # Lab of the Future Workshop - Notebook 1
# MAGIC ## EKG Data Ingestion with Spark Declarative Pipelines
# MAGIC 
# MAGIC This notebook demonstrates how to ingest laboratory instrument data from EKG devices using 
# MAGIC **Spark Declarative Pipelines (formerly Delta Live Tables)** and store it in Unity Catalog.
# MAGIC 
# MAGIC ### Learning Objectives
# MAGIC - Configure a DLT pipeline for streaming EKG data
# MAGIC - Apply data quality constraints and expectations
# MAGIC - Store processed EKG data in Unity Catalog tables
# MAGIC - Monitor data quality metrics
# MAGIC 
# MAGIC ### Architecture
# MAGIC ```
# MAGIC EKG Devices → Raw Landing Zone → Bronze (Raw) → Silver (Cleaned) → Gold (Analytics)
# MAGIC                    ↓                  ↓              ↓               ↓
# MAGIC              UC Volume          UC Table       UC Table        UC Table
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC Set up catalog, schema, and volume paths for the workshop

# COMMAND ----------

# Workshop configuration - modify these for your environment
CATALOG = "lab_of_the_future"
SCHEMA = "healthcare_data"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/raw_ekg_data"

# EKG data configuration
EKG_SAMPLE_RATE_HZ = 500  # Standard 12-lead EKG sampling rate
EKG_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

print(f"Catalog: {CATALOG}")
print(f"Schema: {SCHEMA}")
print(f"Volume Path: {VOLUME_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup: Create Unity Catalog Resources
# MAGIC Create the necessary catalog, schema, and volume if they don't exist

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create catalog for Lab of the Future workshop
# MAGIC CREATE CATALOG IF NOT EXISTS lab_of_the_future
# MAGIC COMMENT 'Lab of the Future Workshop - Healthcare Data Analytics';
# MAGIC 
# MAGIC -- Create schema for healthcare data
# MAGIC CREATE SCHEMA IF NOT EXISTS lab_of_the_future.healthcare_data
# MAGIC COMMENT 'Schema containing EKG, DICOM, and other healthcare instrument data';
# MAGIC 
# MAGIC -- Create volume for raw EKG data landing zone
# MAGIC CREATE VOLUME IF NOT EXISTS lab_of_the_future.healthcare_data.raw_ekg_data
# MAGIC COMMENT 'Landing zone for raw EKG device data files';

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Sample EKG Data
# MAGIC For demonstration purposes, we'll generate synthetic EKG waveform data that simulates real device output

# COMMAND ----------

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os

def generate_ekg_waveform(duration_seconds: int, sample_rate: int, heart_rate_bpm: int = 72):
    """
    Generate a synthetic EKG waveform with realistic P-QRS-T complex patterns.
    """
    num_samples = duration_seconds * sample_rate
    t = np.linspace(0, duration_seconds, num_samples)
    
    # Heart rate determines the period
    period = 60.0 / heart_rate_bpm
    
    # Generate base waveform components
    waveform = np.zeros(num_samples)
    
    for i, t_val in enumerate(t):
        cycle_pos = (t_val % period) / period
        
        # P wave (0.0 - 0.1 of cycle)
        if 0.0 <= cycle_pos < 0.1:
            waveform[i] = 0.25 * np.sin(np.pi * cycle_pos / 0.1)
        
        # QRS complex (0.15 - 0.25 of cycle)
        elif 0.15 <= cycle_pos < 0.17:
            waveform[i] = -0.1 * (cycle_pos - 0.15) / 0.02
        elif 0.17 <= cycle_pos < 0.20:
            waveform[i] = -0.1 + 1.5 * (cycle_pos - 0.17) / 0.03
        elif 0.20 <= cycle_pos < 0.23:
            waveform[i] = 1.4 - 1.6 * (cycle_pos - 0.20) / 0.03
        elif 0.23 <= cycle_pos < 0.25:
            waveform[i] = -0.2 + 0.2 * (cycle_pos - 0.23) / 0.02
        
        # T wave (0.35 - 0.55 of cycle)
        elif 0.35 <= cycle_pos < 0.55:
            waveform[i] = 0.35 * np.sin(np.pi * (cycle_pos - 0.35) / 0.2)
    
    # Add realistic noise
    noise = np.random.normal(0, 0.02, num_samples)
    waveform += noise
    
    return waveform, t

def generate_patient_ekg_record(patient_id: str, recording_time: datetime, duration_seconds: int = 10):
    """
    Generate a complete 12-lead EKG record for a patient.
    """
    heart_rate = np.random.randint(60, 100)  # Normal resting heart rate range
    
    leads_data = {}
    for lead in EKG_LEADS:
        waveform, timestamps = generate_ekg_waveform(duration_seconds, EKG_SAMPLE_RATE_HZ, heart_rate)
        # Add lead-specific variations
        lead_variation = np.random.uniform(0.8, 1.2)
        leads_data[lead] = (waveform * lead_variation).tolist()
    
    return {
        "patient_id": patient_id,
        "device_id": f"EKG-{np.random.randint(1000, 9999)}",
        "recording_timestamp": recording_time.isoformat(),
        "duration_seconds": duration_seconds,
        "sample_rate_hz": EKG_SAMPLE_RATE_HZ,
        "heart_rate_bpm": heart_rate,
        "leads": leads_data,
        "metadata": {
            "device_manufacturer": "MedTech Instruments",
            "device_model": "CardioSync Pro",
            "firmware_version": "2.4.1",
            "calibration_date": (recording_time - timedelta(days=30)).strftime("%Y-%m-%d")
        }
    }

# Generate sample EKG records
print("Generating sample EKG data...")
sample_records = []
base_time = datetime.now()

for i in range(100):
    patient_id = f"PT-{10000 + i}"
    recording_time = base_time - timedelta(hours=np.random.randint(0, 720))  # Past 30 days
    record = generate_patient_ekg_record(patient_id, recording_time)
    sample_records.append(record)

print(f"Generated {len(sample_records)} EKG records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Sample Data to Unity Catalog Volume
# MAGIC Write the generated EKG data to the landing zone volume

# COMMAND ----------

# Save EKG records as JSON files to the volume
import json

# Write individual patient records
for i, record in enumerate(sample_records):
    file_path = f"{VOLUME_PATH}/{record['patient_id']}_{record['recording_timestamp'][:10]}.json"
    
    # Use dbutils to write to volume
    dbutils.fs.put(
        file_path.replace("/Volumes/", "dbfs:/Volumes/"),
        json.dumps(record, indent=2),
        overwrite=True
    )

print(f"Saved {len(sample_records)} EKG records to {VOLUME_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define DLT Pipeline Tables
# MAGIC 
# MAGIC The following cells define the Spark Declarative Pipeline (DLT) tables.
# MAGIC **Note**: These definitions should be run as part of a DLT pipeline, not interactively.
# MAGIC 
# MAGIC To deploy this pipeline:
# MAGIC 1. Create a new DLT pipeline in the Databricks workspace
# MAGIC 2. Point it to this notebook
# MAGIC 3. Configure the target catalog and schema

# COMMAND ----------

# MAGIC %md
# MAGIC ### Interactive Processing: Create Tables Directly
# MAGIC 
# MAGIC For interactive testing, we create the tables directly using Delta Lake.
# MAGIC For production, use the DLT pipeline code shown in the documentation cells below.

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *

# Define the schema for EKG data
ekg_schema = StructType([
    StructField("patient_id", StringType(), False),
    StructField("device_id", StringType(), False),
    StructField("recording_timestamp", StringType(), False),
    StructField("duration_seconds", IntegerType(), False),
    StructField("sample_rate_hz", IntegerType(), False),
    StructField("heart_rate_bpm", IntegerType(), False),
    StructField("leads", MapType(StringType(), ArrayType(DoubleType())), False),
    StructField("metadata", MapType(StringType(), StringType()), True)
])

# Read the JSON files we just wrote
ekg_bronze_df = (
    spark.read
    .format("json")
    .schema(ekg_schema)
    .load(VOLUME_PATH)
    .withColumn("_ingestion_timestamp", current_timestamp())
    .withColumn("_source_file", col("_metadata.file_path"))
)

print(f"Loaded {ekg_bronze_df.count()} EKG records from volume")
ekg_bronze_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Silver Table (Cleaned Data)

# COMMAND ----------

# Apply data quality filters and transformations for silver layer
ekg_silver_df = (
    ekg_bronze_df
    .filter(col("patient_id").isNotNull() & col("patient_id").like("PT-%"))
    .filter(col("heart_rate_bpm").between(30, 250))
    .filter(col("sample_rate_hz") >= 100)
    .withColumn("recording_datetime", to_timestamp("recording_timestamp"))
    .withColumn("recording_date", to_date(col("recording_datetime")))
    .withColumn("device_manufacturer", col("metadata.device_manufacturer"))
    .withColumn("device_model", col("metadata.device_model"))
    .withColumn("firmware_version", col("metadata.firmware_version"))
    .withColumn("lead_count", size(col("leads")))
    .withColumn("_processing_timestamp", current_timestamp())
    .drop("metadata", "recording_timestamp")
)

# Save as Delta table
ekg_silver_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.ekg_silver")
print(f"Created silver table: {CATALOG}.{SCHEMA}.ekg_silver")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Gold Tables (Aggregated Metrics)

# COMMAND ----------

# Patient-level metrics
ekg_patient_metrics = (
    ekg_silver_df
    .groupBy("patient_id")
    .agg(
        count("*").alias("total_recordings"),
        avg("heart_rate_bpm").alias("avg_heart_rate"),
        min("heart_rate_bpm").alias("min_heart_rate"),
        max("heart_rate_bpm").alias("max_heart_rate"),
        stddev("heart_rate_bpm").alias("heart_rate_variability"),
        min("recording_datetime").alias("first_recording"),
        max("recording_datetime").alias("last_recording"),
        collect_set("device_id").alias("devices_used")
    )
)

ekg_patient_metrics.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.ekg_gold_patient_metrics")
print(f"Created gold table: {CATALOG}.{SCHEMA}.ekg_gold_patient_metrics")

# Daily summary
ekg_daily_summary = (
    ekg_silver_df
    .groupBy("recording_date")
    .agg(
        countDistinct("patient_id").alias("unique_patients"),
        count("*").alias("total_recordings"),
        avg("heart_rate_bpm").alias("avg_heart_rate"),
        countDistinct("device_id").alias("active_devices")
    )
    .orderBy("recording_date")
)

ekg_daily_summary.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.ekg_gold_daily_summary")
print(f"Created gold table: {CATALOG}.{SCHEMA}.ekg_gold_daily_summary")

# COMMAND ----------

# MAGIC %md
# MAGIC ## DLT Pipeline Code Reference
# MAGIC 
# MAGIC The code below shows how to define this as a DLT pipeline for production.
# MAGIC **Copy this code to a separate notebook** to run as a DLT pipeline.
# MAGIC 
# MAGIC ```python
# MAGIC import dlt
# MAGIC from pyspark.sql.functions import *
# MAGIC from pyspark.sql.types import *
# MAGIC 
# MAGIC VOLUME_PATH = "/Volumes/lab_of_the_future/healthcare_data/raw_ekg_data"
# MAGIC 
# MAGIC ekg_schema = StructType([
# MAGIC     StructField("patient_id", StringType(), False),
# MAGIC     StructField("device_id", StringType(), False),
# MAGIC     StructField("recording_timestamp", StringType(), False),
# MAGIC     StructField("duration_seconds", IntegerType(), False),
# MAGIC     StructField("sample_rate_hz", IntegerType(), False),
# MAGIC     StructField("heart_rate_bpm", IntegerType(), False),
# MAGIC     StructField("leads", MapType(StringType(), ArrayType(DoubleType())), False),
# MAGIC     StructField("metadata", MapType(StringType(), StringType()), True)
# MAGIC ])
# MAGIC 
# MAGIC @dlt.table(name="ekg_bronze", comment="Raw EKG data")
# MAGIC def ekg_bronze():
# MAGIC     return (spark.readStream.format("cloudFiles")
# MAGIC         .option("cloudFiles.format", "json").schema(ekg_schema)
# MAGIC         .load(VOLUME_PATH))
# MAGIC 
# MAGIC @dlt.table(name="ekg_silver")
# MAGIC @dlt.expect_or_drop("valid_patient_id", "patient_id LIKE 'PT-%'")
# MAGIC @dlt.expect_or_drop("valid_heart_rate", "heart_rate_bpm BETWEEN 30 AND 250")
# MAGIC def ekg_silver():
# MAGIC     return (dlt.read_stream("ekg_bronze")
# MAGIC         .withColumn("recording_datetime", to_timestamp("recording_timestamp")))
# MAGIC 
# MAGIC @dlt.table(name="ekg_gold_patient_metrics")
# MAGIC def ekg_gold_patient_metrics():
# MAGIC     return (dlt.read("ekg_silver").groupBy("patient_id")
# MAGIC         .agg(count("*").alias("total_recordings"),
# MAGIC              avg("heart_rate_bpm").alias("avg_heart_rate")))
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interactive Analysis (Non-DLT)
# MAGIC 
# MAGIC The following cells can be run interactively to explore the data after the DLT pipeline has processed it.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query the Gold Tables

# COMMAND ----------

# MAGIC %sql
# MAGIC -- View patient-level EKG metrics
# MAGIC SELECT 
# MAGIC   patient_id,
# MAGIC   total_recordings,
# MAGIC   ROUND(avg_heart_rate, 1) as avg_heart_rate,
# MAGIC   ROUND(heart_rate_variability, 2) as hrv,
# MAGIC   first_recording,
# MAGIC   last_recording
# MAGIC FROM lab_of_the_future.healthcare_data.ekg_gold_patient_metrics
# MAGIC ORDER BY total_recordings DESC
# MAGIC LIMIT 20;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- View daily summary trends
# MAGIC SELECT 
# MAGIC   recording_date,
# MAGIC   unique_patients,
# MAGIC   total_recordings,
# MAGIC   ROUND(avg_heart_rate, 1) as avg_heart_rate,
# MAGIC   active_devices
# MAGIC FROM lab_of_the_future.healthcare_data.ekg_gold_daily_summary
# MAGIC ORDER BY recording_date DESC
# MAGIC LIMIT 30;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline Monitoring
# MAGIC 
# MAGIC After deploying the DLT pipeline, you can monitor:
# MAGIC - **Data Quality Metrics**: View expectation pass/fail rates in the DLT UI
# MAGIC - **Pipeline Lineage**: Track data flow from bronze to gold layers
# MAGIC - **Processing Latency**: Monitor end-to-end data freshness

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC Continue to **Notebook 2** to learn how to:
# MAGIC - Ingest DICOM medical imaging data
# MAGIC - Store images in Unity Catalog Volumes
# MAGIC - Featurize images using the Databricks Pixels accelerator
# MAGIC 
# MAGIC The EKG data created in this notebook will be combined with imaging data in later notebooks 
# MAGIC to create a comprehensive Lab of the Future digital twin solution.
