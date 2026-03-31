# Databricks notebook source
# MAGIC %md
# MAGIC # Lab of the Future Workshop - Notebook 2
# MAGIC ## DICOM Image Ingestion & Featurization with Databricks Pixels
# MAGIC 
# MAGIC This notebook demonstrates how to use the **[Databricks Pixels Solution Accelerator](https://github.com/databricks-industry-solutions/pixels)** to:
# MAGIC - Ingest DICOM medical imaging data into Unity Catalog Volumes
# MAGIC - Extract metadata from DICOM files at scale
# MAGIC - Catalog and index medical images for SQL access
# MAGIC - Apply metadata anonymization for PHI protection
# MAGIC 
# MAGIC ### What is Pixels?
# MAGIC Pixels is a Databricks Solution Accelerator that facilitates large-scale processing of medical images (DICOM, etc.).
# MAGIC It provides:
# MAGIC - ✅ DICOM metadata extraction and indexing
# MAGIC - ✅ SQL-accessible image catalogs
# MAGIC - ✅ PHI redaction via format-preserving encryption
# MAGIC - ✅ OHIF Viewer integration for visualization
# MAGIC - ✅ MONAI integration for AI segmentation
# MAGIC 
# MAGIC ### Architecture
# MAGIC ```
# MAGIC DICOM Sources → UC Volume → Pixels Catalog → Metadata Extraction → Delta Tables
# MAGIC                    ↓              ↓                  ↓                   ↓
# MAGIC              Raw Files      File Index       DICOM Metadata      SQL/ML Access
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Pixels
# MAGIC 
# MAGIC Install the Pixels package from PyPI or clone from GitHub

# COMMAND ----------

# MAGIC %pip install dbx-pixels pydicom pillow numpy

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Workshop configuration
CATALOG = "lab_of_the_future"
SCHEMA = "healthcare_data"

# Volume paths for DICOM data (using Unity Catalog Volumes)
DICOM_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/dicom_raw"
DICOM_PROCESSED_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/dicom_processed"

# Pixels catalog table
PIXELS_CATALOG_TABLE = f"{CATALOG}.{SCHEMA}.dicom_object_catalog"

print(f"DICOM Volume: {DICOM_VOLUME}")
print(f"Pixels Catalog Table: {PIXELS_CATALOG_TABLE}")

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
# MAGIC -- Create volume for processed/anonymized images
# MAGIC CREATE VOLUME IF NOT EXISTS lab_of_the_future.healthcare_data.dicom_processed
# MAGIC COMMENT 'Processed and anonymized medical images';

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Sample DICOM Files
# MAGIC 
# MAGIC For demonstration, we'll generate synthetic DICOM files using pydicom.
# MAGIC In production, you would ingest real DICOM files from PACS/VNA systems.

# COMMAND ----------

import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
import numpy as np
from datetime import datetime, timedelta
import os
import tempfile

def create_sample_dicom(
    patient_id: str,
    patient_name: str,
    modality: str,
    study_description: str,
    series_description: str,
    instance_number: int = 1,
    image_size: tuple = (512, 512)
):
    """
    Create a sample DICOM file with synthetic image data.
    """
    # Create file meta information
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    
    # Create the FileDataset
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    # Patient information
    ds.PatientID = patient_id
    ds.PatientName = patient_name
    ds.PatientBirthDate = (datetime.now() - timedelta(days=365*np.random.randint(25, 75))).strftime("%Y%m%d")
    ds.PatientSex = np.random.choice(["M", "F"])
    
    # Study information
    ds.StudyInstanceUID = generate_uid()
    ds.StudyDate = (datetime.now() - timedelta(days=np.random.randint(0, 365))).strftime("%Y%m%d")
    ds.StudyTime = f"{np.random.randint(0, 23):02d}{np.random.randint(0, 59):02d}{np.random.randint(0, 59):02d}"
    ds.StudyDescription = study_description
    ds.AccessionNumber = f"ACC{np.random.randint(100000, 999999)}"
    ds.ReferringPhysicianName = f"Dr. Smith-{np.random.randint(1, 20)}"
    
    # Series information
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = np.random.randint(1, 10)
    ds.SeriesDescription = series_description
    ds.Modality = modality
    
    # Instance information
    ds.SOPClassUID = pydicom.uid.CTImageStorage
    ds.SOPInstanceUID = generate_uid()
    ds.InstanceNumber = instance_number
    
    # Equipment information
    ds.Manufacturer = np.random.choice(["GE Healthcare", "Siemens", "Philips", "Canon"])
    ds.InstitutionName = "Lab of the Future Medical Center"
    ds.StationName = f"SCANNER-{np.random.randint(1, 5)}"
    
    # Image information
    ds.Rows = image_size[0]
    ds.Columns = image_size[1]
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    
    # Generate synthetic pixel data based on modality
    if modality == "CT":
        # CT-like data with Hounsfield units simulation
        base = np.random.normal(40, 20, image_size).astype(np.int16)
        # Add circular structures
        for _ in range(np.random.randint(2, 5)):
            cx, cy = np.random.randint(100, image_size[0]-100, 2)
            r = np.random.randint(20, 60)
            y, x = np.ogrid[:image_size[0], :image_size[1]]
            mask = (x - cx)**2 + (y - cy)**2 <= r**2
            base[mask] += np.random.randint(100, 500)
        ds.PixelData = (base + 1024).clip(0, 4095).astype(np.uint16).tobytes()
        
    elif modality == "MR":
        # MRI-like data
        base = np.random.normal(500, 100, image_size).astype(np.int16)
        # Add brain-like ellipse
        cy, cx = image_size[0] // 2, image_size[1] // 2
        y, x = np.ogrid[:image_size[0], :image_size[1]]
        ellipse = ((x - cx) / 150)**2 + ((y - cy) / 180)**2 <= 1
        base[ellipse] += 300
        ds.PixelData = base.clip(0, 4095).astype(np.uint16).tobytes()
        
    elif modality == "XR" or modality == "CR":
        # X-ray like data
        base = np.ones(image_size, dtype=np.int16) * 3000
        # Add bone-like darker regions
        for _ in range(np.random.randint(3, 8)):
            x1 = np.random.randint(50, image_size[1]-50)
            y1 = np.random.randint(50, image_size[0]-50)
            x2 = x1 + np.random.randint(-80, 80)
            y2 = y1 + np.random.randint(-80, 80)
            thickness = np.random.randint(5, 20)
            for t in np.linspace(0, 1, 50):
                px, py = int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))
                if 0 <= px < image_size[1] and 0 <= py < image_size[0]:
                    y_idx, x_idx = np.ogrid[:image_size[0], :image_size[1]]
                    circle = (x_idx - px)**2 + (y_idx - py)**2 <= thickness**2
                    base[circle] = base[circle] * 0.3
        ds.PixelData = base.clip(0, 4095).astype(np.uint16).tobytes()
        
    else:  # US - Ultrasound
        # Ultrasound-like speckle pattern
        base = np.random.exponential(800, image_size).astype(np.int16)
        # Add fan shape
        for y in range(image_size[0]):
            fan_width = int(image_size[1] * 0.3 + (image_size[1] * 0.6 * y / image_size[0]))
            x_start = (image_size[1] - fan_width) // 2
            x_end = x_start + fan_width
            base[y, :x_start] = 0
            base[y, x_end:] = 0
        ds.PixelData = base.clip(0, 4095).astype(np.uint16).tobytes()
    
    # Additional required attributes
    ds.ImagePositionPatient = [0, 0, instance_number * 2.5]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [0.5, 0.5]
    ds.SliceThickness = 2.5
    ds.SliceLocation = instance_number * 2.5
    
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    
    return ds

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Sample DICOM Dataset

# COMMAND ----------

# Generate sample DICOM files
modalities = ["CT", "MR", "XR", "US"]
study_types = {
    "CT": ["CT HEAD", "CT CHEST", "CT ABDOMEN", "CT SPINE"],
    "MR": ["MRI BRAIN", "MRI SPINE", "MRI KNEE", "MRI SHOULDER"],
    "XR": ["CHEST X-RAY", "HAND X-RAY", "SPINE X-RAY", "PELVIS X-RAY"],
    "US": ["ABDOMINAL US", "CARDIAC US", "THYROID US", "PELVIC US"]
}

print("Generating sample DICOM files...")
dicom_files = []

# Generate 50 sample studies with multiple instances
for i in range(20):  # 20 patients
    patient_id = f"PT-{10000 + i}"
    patient_name = f"Patient^Test^{i}"
    
    # Each patient has 1-3 studies
    for study_num in range(np.random.randint(1, 4)):
        modality = np.random.choice(modalities)
        study_desc = np.random.choice(study_types[modality])
        
        # Each study has multiple instances (slices)
        num_instances = np.random.randint(3, 10) if modality in ["CT", "MR"] else np.random.randint(1, 3)
        
        for instance in range(1, num_instances + 1):
            ds = create_sample_dicom(
                patient_id=patient_id,
                patient_name=patient_name,
                modality=modality,
                study_description=study_desc,
                series_description=f"{study_desc} Series",
                instance_number=instance
            )
            dicom_files.append({
                "dataset": ds,
                "patient_id": patient_id,
                "study_uid": ds.StudyInstanceUID,
                "series_uid": ds.SeriesInstanceUID,
                "sop_uid": ds.SOPInstanceUID,
                "modality": modality
            })

print(f"Generated {len(dicom_files)} DICOM instances")
print(f"Modality distribution: {dict(zip(*np.unique([f['modality'] for f in dicom_files], return_counts=True)))}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save DICOM Files to Unity Catalog Volume

# COMMAND ----------

# Save DICOM files to the volume
for dcm_info in dicom_files:
    ds = dcm_info["dataset"]
    patient_id = dcm_info["patient_id"]
    study_uid = dcm_info["study_uid"]
    series_uid = dcm_info["series_uid"]
    sop_uid = dcm_info["sop_uid"]
    
    # Create directory structure: /patient/study/series/
    dir_path = f"/dbfs{DICOM_VOLUME}/{patient_id}/{study_uid[:20]}/{series_uid[:20]}"
    os.makedirs(dir_path, exist_ok=True)
    
    # Save the DICOM file
    file_path = f"{dir_path}/{sop_uid[:20]}.dcm"
    ds.save_as(file_path)

print(f"Saved {len(dicom_files)} DICOM files to {DICOM_VOLUME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize Databricks Pixels
# MAGIC 
# MAGIC Use the Pixels library to catalog and extract metadata from DICOM files.

# COMMAND ----------

from dbx.pixels import Catalog
from dbx.pixels.dicom import DicomMetaExtractor

# Initialize the Pixels Catalog
catalog = Catalog(spark)

print("Pixels Catalog initialized")
print(f"Scanning volume: {DICOM_VOLUME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Catalog DICOM Files
# MAGIC 
# MAGIC The Pixels `Catalog` object indexes all files in the specified path.

# COMMAND ----------

# Catalog all DICOM files in the volume
# This creates an index of all files with their paths and basic metadata
catalog_df = catalog.catalog(DICOM_VOLUME)

print(f"Cataloged files:")
display(catalog_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract DICOM Metadata
# MAGIC 
# MAGIC Use `DicomMetaExtractor` to parse and extract all DICOM tags from each file.

# COMMAND ----------

# Extract DICOM metadata from all cataloged files
# This parses each DICOM file and extracts all metadata tags
meta_df = DicomMetaExtractor(catalog).transform(catalog_df)

print(f"Extracted metadata from {meta_df.count()} DICOM files")
meta_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Catalog to Delta Table
# MAGIC 
# MAGIC Save the extracted metadata to a Delta table for SQL access.

# COMMAND ----------

# Save the metadata catalog to Unity Catalog
# This creates a Delta table with all DICOM metadata
catalog.save(meta_df, path=PIXELS_CATALOG_TABLE)

print(f"Saved Pixels catalog to: {PIXELS_CATALOG_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query the DICOM Catalog with SQL
# MAGIC 
# MAGIC Once saved, you can use SQL to analyze your medical imaging data.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- View the cataloged DICOM metadata
# MAGIC SELECT 
# MAGIC   path,
# MAGIC   meta:PatientID::STRING as patient_id,
# MAGIC   meta:PatientName::STRING as patient_name,
# MAGIC   meta:Modality::STRING as modality,
# MAGIC   meta:StudyDescription::STRING as study_description,
# MAGIC   meta:SeriesDescription::STRING as series_description,
# MAGIC   meta:Manufacturer::STRING as manufacturer,
# MAGIC   meta:StudyDate::STRING as study_date,
# MAGIC   meta:Rows::INT as rows,
# MAGIC   meta:Columns::INT as columns
# MAGIC FROM lab_of_the_future.healthcare_data.dicom_object_catalog
# MAGIC LIMIT 20;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Analyze imaging studies by modality
# MAGIC SELECT 
# MAGIC   meta:Modality::STRING as modality,
# MAGIC   COUNT(*) as instance_count,
# MAGIC   COUNT(DISTINCT meta:PatientID::STRING) as unique_patients,
# MAGIC   COUNT(DISTINCT meta:StudyInstanceUID::STRING) as unique_studies
# MAGIC FROM lab_of_the_future.healthcare_data.dicom_object_catalog
# MAGIC GROUP BY meta:Modality::STRING
# MAGIC ORDER BY instance_count DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Study volume by manufacturer
# MAGIC SELECT 
# MAGIC   meta:Manufacturer::STRING as manufacturer,
# MAGIC   meta:Modality::STRING as modality,
# MAGIC   COUNT(*) as image_count
# MAGIC FROM lab_of_the_future.healthcare_data.dicom_object_catalog
# MAGIC GROUP BY meta:Manufacturer::STRING, meta:Modality::STRING
# MAGIC ORDER BY image_count DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Incremental Processing with Auto Loader
# MAGIC 
# MAGIC For production workloads, use streaming/incremental processing to handle new files.

# COMMAND ----------

# Example: Incremental processing with Auto Loader
# This enables streaming ingestion of new DICOM files

# checkpoint_path = f"/Volumes/{CATALOG}/{SCHEMA}/checkpoints/dicom_catalog"
# 
# catalog_df_streaming = catalog.catalog(
#     DICOM_VOLUME, 
#     streaming=True, 
#     streamCheckpointBasePath=checkpoint_path
# )
# 
# meta_df_streaming = DicomMetaExtractor(catalog).transform(catalog_df_streaming)
# 
# # Write as a streaming table
# meta_df_streaming.writeStream \
#     .format("delta") \
#     .outputMode("append") \
#     .option("checkpointLocation", f"{checkpoint_path}/meta") \
#     .toTable(PIXELS_CATALOG_TABLE)

print("Streaming example code provided (not executed)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Metadata Anonymization (PHI Protection)
# MAGIC 
# MAGIC Pixels supports format-preserving encryption for HIPAA compliance.

# COMMAND ----------

# Example: Anonymize DICOM metadata for PHI protection
# 
# from dbx.pixels.dicom import DicomMetaAnonymizerExtractor
# 
# # Format-preserving encryption key (128, 192, or 256 bits - hex string)
# fp_key = "2B7E151628AED2A6ABF7158809CF4F3C"  # Example - use secure key in production
# fp_tweak = "A1B2C3D4E5F60718"  # 64-bit tweak
# 
# anonymized_df = DicomMetaAnonymizerExtractor(
#     catalog,
#     anonym_mode="METADATA",
#     fp_key=fp_key,
#     fp_tweak=fp_tweak,
#     anonymization_base_path=DICOM_PROCESSED_VOLUME
# ).transform(catalog_df)
# 
# catalog.save(anonymized_df, path=f"{CATALOG}.{SCHEMA}.dicom_anonymized_catalog")

print("Anonymization example code provided (not executed)")
print("In production, use secure keys stored in Databricks Secrets")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Feature Views for ML
# MAGIC 
# MAGIC Create views that extract features useful for machine learning.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create a feature view for ML models
# MAGIC CREATE OR REPLACE VIEW lab_of_the_future.healthcare_data.dicom_ml_features AS
# MAGIC SELECT 
# MAGIC   path,
# MAGIC   meta:PatientID::STRING as patient_id,
# MAGIC   meta:StudyInstanceUID::STRING as study_uid,
# MAGIC   meta:SeriesInstanceUID::STRING as series_uid,
# MAGIC   meta:SOPInstanceUID::STRING as sop_uid,
# MAGIC   meta:Modality::STRING as modality,
# MAGIC   meta:StudyDescription::STRING as study_description,
# MAGIC   meta:BodyPartExamined::STRING as body_part,
# MAGIC   meta:Rows::INT as image_rows,
# MAGIC   meta:Columns::INT as image_cols,
# MAGIC   meta:BitsAllocated::INT as bits_allocated,
# MAGIC   meta:PixelSpacing::STRING as pixel_spacing,
# MAGIC   meta:SliceThickness::DOUBLE as slice_thickness,
# MAGIC   meta:Manufacturer::STRING as manufacturer,
# MAGIC   meta:InstitutionName::STRING as institution,
# MAGIC   -- Calculate derived features
# MAGIC   meta:Rows::INT * meta:Columns::INT as total_pixels,
# MAGIC   CASE 
# MAGIC     WHEN meta:Modality::STRING = 'CT' THEN 'volumetric'
# MAGIC     WHEN meta:Modality::STRING = 'MR' THEN 'volumetric'
# MAGIC     ELSE 'planar'
# MAGIC   END as image_type,
# MAGIC   -- Timestamp for tracking
# MAGIC   current_timestamp() as feature_extraction_time
# MAGIC FROM lab_of_the_future.healthcare_data.dicom_object_catalog;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Preview the ML features view
# MAGIC SELECT * FROM lab_of_the_future.healthcare_data.dicom_ml_features LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Integration with OHIF Viewer
# MAGIC 
# MAGIC Pixels includes a pre-built OHIF Viewer integration for visualizing DICOM images.
# MAGIC 
# MAGIC To use the OHIF Viewer:
# MAGIC 1. Run the `06-OHIF-Viewer` notebook from the Pixels repository
# MAGIC 2. Set the `table` parameter to your Pixels catalog table
# MAGIC 3. Set the `sqlWarehouseID` to your SQL Warehouse ID
# MAGIC 
# MAGIC The viewer provides:
# MAGIC - Interactive study list from your catalog
# MAGIC - Multi-layer visualization for CT/MRI
# MAGIC - Measurement and annotation tools
# MAGIC - Segmentation export capabilities

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary Statistics

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Summary of ingested DICOM data
# MAGIC SELECT 
# MAGIC   'Total DICOM Files' as metric,
# MAGIC   CAST(COUNT(*) AS STRING) as value
# MAGIC FROM lab_of_the_future.healthcare_data.dicom_object_catalog
# MAGIC UNION ALL
# MAGIC SELECT 
# MAGIC   'Unique Patients',
# MAGIC   CAST(COUNT(DISTINCT meta:PatientID::STRING) AS STRING)
# MAGIC FROM lab_of_the_future.healthcare_data.dicom_object_catalog
# MAGIC UNION ALL
# MAGIC SELECT 
# MAGIC   'Unique Studies',
# MAGIC   CAST(COUNT(DISTINCT meta:StudyInstanceUID::STRING) AS STRING)
# MAGIC FROM lab_of_the_future.healthcare_data.dicom_object_catalog
# MAGIC UNION ALL
# MAGIC SELECT 
# MAGIC   'Unique Series',
# MAGIC   CAST(COUNT(DISTINCT meta:SeriesInstanceUID::STRING) AS STRING)
# MAGIC FROM lab_of_the_future.healthcare_data.dicom_object_catalog
# MAGIC UNION ALL
# MAGIC SELECT 
# MAGIC   'Modalities',
# MAGIC   CAST(COUNT(DISTINCT meta:Modality::STRING) AS STRING)
# MAGIC FROM lab_of_the_future.healthcare_data.dicom_object_catalog;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC Continue to **Notebook 3** to learn how to:
# MAGIC - Build a surgery room digital twin prototype
# MAGIC - Combine EKG data from Notebook 1 with DICOM images
# MAGIC - Integrate real-time video data from operating rooms
# MAGIC - Visualize multi-modal healthcare data
# MAGIC 
# MAGIC ### Additional Resources
# MAGIC 
# MAGIC - [Pixels GitHub Repository](https://github.com/databricks-industry-solutions/pixels)
# MAGIC - [OHIF Viewer Documentation](https://docs.ohif.org/)
# MAGIC - [MONAI Medical AI Framework](https://monai.io/)
# MAGIC - [pydicom Documentation](https://pydicom.github.io/)
