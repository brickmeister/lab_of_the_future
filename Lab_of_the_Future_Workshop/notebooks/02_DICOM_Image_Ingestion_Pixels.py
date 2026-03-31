# Databricks notebook source
# MAGIC %md
# MAGIC # Lab of the Future Workshop - Notebook 2
# MAGIC ## DICOM Image Ingestion & Featurization with Databricks Pixels
# MAGIC 
# MAGIC This notebook demonstrates how to:
# MAGIC - Ingest DICOM medical imaging data into Unity Catalog Volumes
# MAGIC - Extract metadata and pixel data from DICOM files
# MAGIC - Featurize images using the **Databricks Pixels Accelerator**
# MAGIC - Store featurized data for downstream ML and analytics
# MAGIC 
# MAGIC ### Learning Objectives
# MAGIC - Understand DICOM file structure and metadata
# MAGIC - Configure Unity Catalog Volumes for medical image storage
# MAGIC - Use Pixels accelerator for distributed image processing
# MAGIC - Extract embeddings and features for ML models
# MAGIC 
# MAGIC ### Architecture
# MAGIC ```
# MAGIC DICOM Sources → UC Volume (Raw) → Pixels Processing → Feature Tables (UC)
# MAGIC                      ↓                    ↓                   ↓
# MAGIC                 Raw Images         Metadata + Pixels    Embeddings + Features
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies
# MAGIC 
# MAGIC Install the Databricks Pixels library and DICOM processing packages

# COMMAND ----------

# MAGIC %pip install databricks-pixels pydicom pillow numpy opencv-python-headless

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Workshop configuration
CATALOG = "lab_of_the_future"
SCHEMA = "healthcare_data"

# Volume paths for DICOM data
DICOM_RAW_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/dicom_raw"
DICOM_PROCESSED_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/dicom_processed"

# Tables for metadata and features
DICOM_METADATA_TABLE = f"{CATALOG}.{SCHEMA}.dicom_metadata"
DICOM_FEATURES_TABLE = f"{CATALOG}.{SCHEMA}.dicom_features"

print(f"Raw DICOM Volume: {DICOM_RAW_VOLUME}")
print(f"Processed Volume: {DICOM_PROCESSED_VOLUME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup: Create Unity Catalog Resources

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Ensure catalog and schema exist (created in Notebook 1)
# MAGIC CREATE CATALOG IF NOT EXISTS lab_of_the_future;
# MAGIC CREATE SCHEMA IF NOT EXISTS lab_of_the_future.healthcare_data;
# MAGIC 
# MAGIC -- Create volume for raw DICOM files
# MAGIC CREATE VOLUME IF NOT EXISTS lab_of_the_future.healthcare_data.dicom_raw
# MAGIC COMMENT 'Raw DICOM medical imaging files from various modalities';
# MAGIC 
# MAGIC -- Create volume for processed images
# MAGIC CREATE VOLUME IF NOT EXISTS lab_of_the_future.healthcare_data.dicom_processed
# MAGIC COMMENT 'Processed and normalized medical images';

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Sample DICOM-like Data
# MAGIC 
# MAGIC For demonstration, we'll generate synthetic medical image data that simulates DICOM files.
# MAGIC In production, you would ingest real DICOM files from PACS systems or imaging devices.

# COMMAND ----------

import numpy as np
from PIL import Image
import io
import json
from datetime import datetime, timedelta
import os

def generate_synthetic_medical_image(modality: str, width: int = 512, height: int = 512):
    """
    Generate a synthetic medical image for demonstration.
    """
    if modality == "CT":
        # CT-like grayscale with tissue-like patterns
        base = np.random.normal(128, 30, (height, width)).astype(np.uint8)
        # Add circular structures (simulating organs/bones)
        for _ in range(np.random.randint(2, 5)):
            cx, cy = np.random.randint(100, width-100), np.random.randint(100, height-100)
            r = np.random.randint(30, 80)
            y, x = np.ogrid[:height, :width]
            mask = (x - cx)**2 + (y - cy)**2 <= r**2
            base[mask] = np.clip(base[mask] + np.random.randint(30, 100), 0, 255)
        return base
    
    elif modality == "MRI":
        # MRI-like with different contrast
        base = np.random.normal(100, 20, (height, width)).astype(np.uint8)
        # Add brain-like elliptical structure
        cy, cx = height // 2, width // 2
        y, x = np.ogrid[:height, :width]
        ellipse = ((x - cx) / 180)**2 + ((y - cy) / 220)**2 <= 1
        base[ellipse] = np.clip(base[ellipse] + 80, 0, 255)
        # Add ventricle-like dark regions
        for _ in range(2):
            vx = cx + np.random.randint(-50, 50)
            vy = cy + np.random.randint(-30, 30)
            vr = np.random.randint(15, 30)
            vmask = (x - vx)**2 + (y - vy)**2 <= vr**2
            base[vmask] = np.clip(base[vmask] - 60, 0, 255)
        return base
    
    elif modality == "XR":
        # X-ray like image
        base = np.ones((height, width), dtype=np.uint8) * 200
        # Add bone-like structures
        for _ in range(np.random.randint(3, 7)):
            x1, y1 = np.random.randint(50, width-50), np.random.randint(50, height-50)
            x2, y2 = x1 + np.random.randint(-100, 100), y1 + np.random.randint(-100, 100)
            thickness = np.random.randint(10, 30)
            # Draw line-like structure
            for t in np.linspace(0, 1, 100):
                px, py = int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))
                if 0 <= px < width and 0 <= py < height:
                    y_idx, x_idx = np.ogrid[:height, :width]
                    circle = (x_idx - px)**2 + (y_idx - py)**2 <= (thickness//2)**2
                    base[circle] = np.clip(base[circle] - 100, 0, 255)
        return base
    
    elif modality == "US":
        # Ultrasound-like with speckle noise
        base = np.random.exponential(50, (height, width)).astype(np.uint8)
        # Add fan shape
        cy = 0
        for y in range(height):
            fan_width = int(width * 0.3 + (width * 0.7 * y / height))
            x_start = (width - fan_width) // 2
            x_end = x_start + fan_width
            base[y, :x_start] = 0
            base[y, x_end:] = 0
        # Add speckle
        speckle = np.random.rayleigh(30, (height, width))
        base = np.clip(base + speckle, 0, 255).astype(np.uint8)
        return base
    
    return np.random.randint(0, 255, (height, width), dtype=np.uint8)

def generate_dicom_metadata(patient_id: str, study_id: str, modality: str, acquisition_time: datetime):
    """
    Generate DICOM-like metadata structure.
    """
    series_id = f"SE-{np.random.randint(100000, 999999)}"
    instance_id = f"IN-{np.random.randint(100000, 999999)}"
    
    return {
        "patient_id": patient_id,
        "patient_name": f"Patient_{patient_id}",
        "patient_birth_date": (datetime.now() - timedelta(days=365*np.random.randint(20, 80))).strftime("%Y%m%d"),
        "patient_sex": np.random.choice(["M", "F"]),
        "study_instance_uid": study_id,
        "study_date": acquisition_time.strftime("%Y%m%d"),
        "study_time": acquisition_time.strftime("%H%M%S"),
        "study_description": f"{modality} Study",
        "series_instance_uid": series_id,
        "series_description": f"{modality} Series",
        "series_number": np.random.randint(1, 10),
        "modality": modality,
        "sop_instance_uid": instance_id,
        "instance_number": np.random.randint(1, 100),
        "rows": 512,
        "columns": 512,
        "bits_allocated": 8,
        "bits_stored": 8,
        "pixel_spacing": [0.5, 0.5],
        "slice_thickness": 2.5,
        "manufacturer": np.random.choice(["GE Healthcare", "Siemens", "Philips", "Canon"]),
        "institution_name": "Lab of the Future Medical Center",
        "acquisition_datetime": acquisition_time.isoformat(),
        "body_part_examined": np.random.choice(["HEAD", "CHEST", "ABDOMEN", "EXTREMITY"]),
        "referring_physician": f"Dr. Smith-{np.random.randint(1, 20)}"
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate and Store Sample Images

# COMMAND ----------

# Generate sample medical images
modalities = ["CT", "MRI", "XR", "US"]
sample_images = []
base_time = datetime.now()

print("Generating sample medical images...")

for i in range(50):  # Generate 50 sample images
    patient_id = f"PT-{10000 + (i % 20)}"  # 20 unique patients
    study_id = f"ST-{np.random.randint(100000, 999999)}"
    modality = np.random.choice(modalities)
    acq_time = base_time - timedelta(days=np.random.randint(0, 90))
    
    # Generate image
    pixel_array = generate_synthetic_medical_image(modality)
    
    # Generate metadata
    metadata = generate_dicom_metadata(patient_id, study_id, modality, acq_time)
    
    sample_images.append({
        "metadata": metadata,
        "pixel_array": pixel_array
    })

print(f"Generated {len(sample_images)} sample images")
print(f"Modality distribution: {dict(zip(*np.unique([img['metadata']['modality'] for img in sample_images], return_counts=True)))}")

# COMMAND ----------

# Save images and metadata to volume
from PIL import Image
import json

for i, sample in enumerate(sample_images):
    metadata = sample["metadata"]
    pixel_array = sample["pixel_array"]
    
    # Create file paths
    patient_folder = f"{DICOM_RAW_VOLUME}/{metadata['patient_id']}/{metadata['study_instance_uid']}"
    image_filename = f"{metadata['sop_instance_uid']}.png"
    metadata_filename = f"{metadata['sop_instance_uid']}.json"
    
    # Convert to PIL Image and save
    img = Image.fromarray(pixel_array, mode='L')
    
    # Save image using dbutils
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_bytes = img_buffer.getvalue()
    
    # Write files to volume
    dbutils.fs.mkdirs(patient_folder.replace("/Volumes/", "dbfs:/Volumes/"))
    
    with open(f"/dbfs{patient_folder}/{image_filename}", 'wb') as f:
        f.write(img_bytes)
    
    with open(f"/dbfs{patient_folder}/{metadata_filename}", 'w') as f:
        json.dump(metadata, f, indent=2)

print(f"Saved {len(sample_images)} images and metadata files to {DICOM_RAW_VOLUME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize Databricks Pixels
# MAGIC 
# MAGIC The Pixels accelerator provides distributed image processing capabilities optimized for medical imaging.

# COMMAND ----------

from databricks.pixels import ImageSchema, read_image
from pyspark.sql.functions import *
from pyspark.sql.types import *

# COMMAND ----------

# MAGIC %md
# MAGIC ## Catalog DICOM Files
# MAGIC 
# MAGIC Use Pixels to catalog all image files in the volume

# COMMAND ----------

# Read all image files from the volume using Pixels
image_catalog_df = (
    spark.read
    .format("binaryFile")
    .option("pathGlobFilter", "*.png")
    .option("recursiveFileLookup", "true")
    .load(DICOM_RAW_VOLUME)
)

# Extract path components for organization
image_catalog_df = (
    image_catalog_df
    .withColumn("file_path", col("path"))
    .withColumn("file_name", regexp_extract(col("path"), r"/([^/]+)\.png$", 1))
    .withColumn("patient_id", regexp_extract(col("path"), r"/(PT-\d+)/", 1))
    .withColumn("study_uid", regexp_extract(col("path"), r"/(ST-\d+)/", 1))
    .withColumn("file_size_kb", round(col("length") / 1024, 2))
    .withColumn("catalog_timestamp", current_timestamp())
)

display(image_catalog_df.select("file_path", "patient_id", "study_uid", "file_size_kb", "catalog_timestamp").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and Parse DICOM Metadata
# MAGIC 
# MAGIC Load the JSON metadata files alongside the images

# COMMAND ----------

# Read metadata JSON files
metadata_df = (
    spark.read
    .format("json")
    .option("recursiveFileLookup", "true")
    .load(f"{DICOM_RAW_VOLUME}/*/*.json")
)

# Display metadata schema
print("DICOM Metadata Schema:")
metadata_df.printSchema()

# COMMAND ----------

# Store metadata in Unity Catalog table
metadata_df.write.mode("overwrite").saveAsTable(DICOM_METADATA_TABLE)

print(f"Metadata saved to {DICOM_METADATA_TABLE}")
display(spark.sql(f"SELECT * FROM {DICOM_METADATA_TABLE} LIMIT 10"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Image Featurization with Pixels
# MAGIC 
# MAGIC Extract features from medical images using the Pixels accelerator

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType, StructType, StructField
import numpy as np
from PIL import Image
import io

# Define feature extraction functions
def extract_image_features(image_bytes):
    """
    Extract basic statistical and texture features from a medical image.
    """
    try:
        # Load image
        img = Image.open(io.BytesIO(image_bytes))
        pixel_array = np.array(img)
        
        # Basic statistics
        features = {
            "mean_intensity": float(np.mean(pixel_array)),
            "std_intensity": float(np.std(pixel_array)),
            "min_intensity": float(np.min(pixel_array)),
            "max_intensity": float(np.max(pixel_array)),
            "median_intensity": float(np.median(pixel_array)),
            "intensity_range": float(np.max(pixel_array) - np.min(pixel_array)),
        }
        
        # Histogram features (10 bins)
        hist, _ = np.histogram(pixel_array.flatten(), bins=10, range=(0, 255))
        hist_normalized = hist / hist.sum()
        features["histogram_entropy"] = float(-np.sum(hist_normalized * np.log2(hist_normalized + 1e-10)))
        
        # Texture features (simple gradient-based)
        if len(pixel_array.shape) == 2:
            gx = np.gradient(pixel_array.astype(float), axis=1)
            gy = np.gradient(pixel_array.astype(float), axis=0)
            gradient_magnitude = np.sqrt(gx**2 + gy**2)
            features["gradient_mean"] = float(np.mean(gradient_magnitude))
            features["gradient_std"] = float(np.std(gradient_magnitude))
        else:
            features["gradient_mean"] = 0.0
            features["gradient_std"] = 0.0
        
        # Edge density (simple threshold on gradient)
        edge_threshold = np.percentile(gradient_magnitude, 90)
        edge_mask = gradient_magnitude > edge_threshold
        features["edge_density"] = float(np.mean(edge_mask))
        
        # Quadrant analysis
        h, w = pixel_array.shape[:2]
        quadrants = [
            pixel_array[:h//2, :w//2],
            pixel_array[:h//2, w//2:],
            pixel_array[h//2:, :w//2],
            pixel_array[h//2:, w//2:]
        ]
        features["quadrant_variance"] = float(np.var([np.mean(q) for q in quadrants]))
        
        return features
    except Exception as e:
        return None

# Register UDF
feature_schema = StructType([
    StructField("mean_intensity", FloatType()),
    StructField("std_intensity", FloatType()),
    StructField("min_intensity", FloatType()),
    StructField("max_intensity", FloatType()),
    StructField("median_intensity", FloatType()),
    StructField("intensity_range", FloatType()),
    StructField("histogram_entropy", FloatType()),
    StructField("gradient_mean", FloatType()),
    StructField("gradient_std", FloatType()),
    StructField("edge_density", FloatType()),
    StructField("quadrant_variance", FloatType())
])

extract_features_udf = udf(extract_image_features, feature_schema)

# COMMAND ----------

# Apply feature extraction to all images
features_df = (
    image_catalog_df
    .withColumn("features", extract_features_udf(col("content")))
    .select(
        "file_path",
        "patient_id",
        "study_uid",
        "file_name",
        "file_size_kb",
        "features.*",
        "catalog_timestamp"
    )
    .filter(col("mean_intensity").isNotNull())
)

# Display feature results
display(features_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Store Features in Unity Catalog

# COMMAND ----------

# Save features to Unity Catalog table
features_df.write.mode("overwrite").saveAsTable(DICOM_FEATURES_TABLE)

print(f"Features saved to {DICOM_FEATURES_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Join Features with Metadata

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create a comprehensive view joining features with metadata
# MAGIC CREATE OR REPLACE VIEW lab_of_the_future.healthcare_data.dicom_complete AS
# MAGIC SELECT 
# MAGIC   f.file_path,
# MAGIC   f.patient_id,
# MAGIC   f.study_uid,
# MAGIC   m.modality,
# MAGIC   m.body_part_examined,
# MAGIC   m.study_description,
# MAGIC   m.manufacturer,
# MAGIC   m.acquisition_datetime,
# MAGIC   f.mean_intensity,
# MAGIC   f.std_intensity,
# MAGIC   f.histogram_entropy,
# MAGIC   f.gradient_mean,
# MAGIC   f.gradient_std,
# MAGIC   f.edge_density,
# MAGIC   f.quadrant_variance,
# MAGIC   f.file_size_kb,
# MAGIC   f.catalog_timestamp
# MAGIC FROM lab_of_the_future.healthcare_data.dicom_features f
# MAGIC LEFT JOIN lab_of_the_future.healthcare_data.dicom_metadata m
# MAGIC   ON f.file_name = m.sop_instance_uid;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Explore the complete dataset
# MAGIC SELECT 
# MAGIC   modality,
# MAGIC   COUNT(*) as image_count,
# MAGIC   ROUND(AVG(mean_intensity), 2) as avg_intensity,
# MAGIC   ROUND(AVG(histogram_entropy), 2) as avg_entropy,
# MAGIC   ROUND(AVG(edge_density), 4) as avg_edge_density
# MAGIC FROM lab_of_the_future.healthcare_data.dicom_complete
# MAGIC GROUP BY modality
# MAGIC ORDER BY image_count DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced: Generate Image Embeddings
# MAGIC 
# MAGIC For ML use cases, we can generate embedding vectors using pre-trained models

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType

def generate_simple_embedding(image_bytes, embedding_dim=128):
    """
    Generate a simple embedding vector from image features.
    In production, this would use a pre-trained model like ResNet or a medical imaging model.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        pixel_array = np.array(img).astype(float)
        
        # Resize to fixed size for consistent embedding
        img_resized = img.resize((16, 16))
        flat = np.array(img_resized).flatten()
        
        # Normalize and pad/truncate to embedding dimension
        embedding = np.zeros(embedding_dim)
        flat_normalized = (flat - flat.mean()) / (flat.std() + 1e-10)
        embedding[:min(len(flat_normalized), embedding_dim)] = flat_normalized[:embedding_dim]
        
        return embedding.tolist()
    except Exception:
        return [0.0] * embedding_dim

embedding_udf = udf(generate_simple_embedding, ArrayType(FloatType()))

# COMMAND ----------

# Generate embeddings for all images
embeddings_df = (
    image_catalog_df
    .withColumn("embedding", embedding_udf(col("content")))
    .select("file_path", "patient_id", "study_uid", "file_name", "embedding")
)

# Save embeddings
embeddings_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.dicom_embeddings")

print(f"Embeddings saved to {CATALOG}.{SCHEMA}.dicom_embeddings")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary Statistics

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Summary of ingested DICOM data
# MAGIC SELECT 
# MAGIC   'Total Images' as metric,
# MAGIC   CAST(COUNT(*) AS STRING) as value
# MAGIC FROM lab_of_the_future.healthcare_data.dicom_features
# MAGIC UNION ALL
# MAGIC SELECT 
# MAGIC   'Unique Patients',
# MAGIC   CAST(COUNT(DISTINCT patient_id) AS STRING)
# MAGIC FROM lab_of_the_future.healthcare_data.dicom_features
# MAGIC UNION ALL
# MAGIC SELECT 
# MAGIC   'Unique Studies',
# MAGIC   CAST(COUNT(DISTINCT study_uid) AS STRING)
# MAGIC FROM lab_of_the_future.healthcare_data.dicom_features
# MAGIC UNION ALL
# MAGIC SELECT
# MAGIC   'Total Size (MB)',
# MAGIC   CAST(ROUND(SUM(file_size_kb) / 1024, 2) AS STRING)
# MAGIC FROM lab_of_the_future.healthcare_data.dicom_features;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC Continue to **Notebook 3** to learn how to:
# MAGIC - Build a surgery room digital twin prototype
# MAGIC - Combine EKG data from Notebook 1 with DICOM images
# MAGIC - Integrate real-time video data from operating rooms
# MAGIC - Visualize multi-modal healthcare data in a unified dashboard
