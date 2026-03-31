# Databricks notebook source
# MAGIC %md
# MAGIC # Lab of the Future Workshop - Notebook 2a
# MAGIC ## DICOM Medical Image Data Generation
# MAGIC 
# MAGIC This notebook generates synthetic DICOM medical imaging data for the workshop.
# MAGIC 
# MAGIC ### What is DICOM?
# MAGIC **DICOM (Digital Imaging and Communications in Medicine)** is the international standard for medical images.
# MAGIC It defines formats for medical images that can be exchanged with the data and quality necessary for clinical use.
# MAGIC 
# MAGIC DICOM is implemented in virtually every:
# MAGIC - Radiology device (X-ray, CT, MRI, Ultrasound)
# MAGIC - Cardiology imaging system
# MAGIC - Radiotherapy device
# MAGIC - Ophthalmology and dentistry equipment
# MAGIC 
# MAGIC ### What This Notebook Does
# MAGIC 1. Creates Unity Catalog resources (volumes) for storing medical images
# MAGIC 2. Generates synthetic DICOM files with realistic metadata for multiple modalities:
# MAGIC    - **CT** (Computed Tomography)
# MAGIC    - **MR** (Magnetic Resonance Imaging)
# MAGIC    - **XR** (X-Ray / Radiography)
# MAGIC    - **US** (Ultrasound)
# MAGIC 3. Saves DICOM files to Unity Catalog Volumes with proper DICOM hierarchy
# MAGIC 4. Creates a metadata catalog table for downstream processing
# MAGIC 
# MAGIC ### DICOM Data Hierarchy
# MAGIC ```
# MAGIC Patient
# MAGIC   └── Study (exam/visit)
# MAGIC         └── Series (image sequence)
# MAGIC               └── Instance (individual image/slice)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies

# COMMAND ----------

# MAGIC %pip install pydicom pillow numpy --quiet

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
DICOM_VOLUME_PATH = f"{DICOM_VOLUME}"

# Metadata table for tracking generated DICOM files
DICOM_METADATA_TABLE = f"{CATALOG}.{SCHEMA}.dicom_metadata"

print(f"Catalog: {CATALOG}")
print(f"Schema: {SCHEMA}")
print(f"DICOM Volume: {DICOM_VOLUME}")
print(f"Metadata Table: {DICOM_METADATA_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup: Create Unity Catalog Resources

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create catalog and schema
# MAGIC CREATE CATALOG IF NOT EXISTS lab_of_the_future;
# MAGIC CREATE SCHEMA IF NOT EXISTS lab_of_the_future.healthcare_data;
# MAGIC 
# MAGIC -- Create volume for raw DICOM files
# MAGIC CREATE VOLUME IF NOT EXISTS lab_of_the_future.healthcare_data.dicom_raw
# MAGIC COMMENT 'Raw DICOM medical imaging files from various modalities (CT, MR, XR, US)';
# MAGIC 
# MAGIC -- Create volume for processed/anonymized images
# MAGIC CREATE VOLUME IF NOT EXISTS lab_of_the_future.healthcare_data.dicom_processed
# MAGIC COMMENT 'Processed and de-identified medical images';

# COMMAND ----------

# MAGIC %md
# MAGIC ## DICOM File Generator
# MAGIC 
# MAGIC This section defines functions to generate synthetic but realistic DICOM files
# MAGIC with proper metadata conforming to the DICOM standard.

# COMMAND ----------

import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
from pydicom.sequence import Sequence
import numpy as np
from datetime import datetime, timedelta
import os
import json

# DICOM SOP Class UIDs for different modalities
SOP_CLASS_UIDS = {
    "CT": pydicom.uid.CTImageStorage,
    "MR": pydicom.uid.MRImageStorage,
    "XR": pydicom.uid.DigitalXRayImageStorageForPresentation,
    "CR": pydicom.uid.ComputedRadiographyImageStorage,
    "US": pydicom.uid.UltrasoundImageStorage
}

# Realistic study descriptions by modality
STUDY_DESCRIPTIONS = {
    "CT": [
        "CT HEAD WITHOUT CONTRAST",
        "CT HEAD WITH AND WITHOUT CONTRAST", 
        "CT CHEST WITH CONTRAST",
        "CT CHEST WITHOUT CONTRAST",
        "CT ABDOMEN AND PELVIS WITH CONTRAST",
        "CT SPINE CERVICAL WITHOUT CONTRAST",
        "CT SPINE LUMBAR WITHOUT CONTRAST",
        "CT ANGIOGRAM CHEST",
        "CT CARDIAC CALCIUM SCORING"
    ],
    "MR": [
        "MRI BRAIN WITHOUT CONTRAST",
        "MRI BRAIN WITH AND WITHOUT CONTRAST",
        "MRI SPINE CERVICAL WITHOUT CONTRAST",
        "MRI SPINE LUMBAR WITHOUT CONTRAST",
        "MRI KNEE LEFT WITHOUT CONTRAST",
        "MRI KNEE RIGHT WITHOUT CONTRAST",
        "MRI SHOULDER LEFT WITHOUT CONTRAST",
        "MRI CARDIAC FUNCTION",
        "MRI ABDOMEN WITH CONTRAST"
    ],
    "XR": [
        "XR CHEST 2 VIEWS",
        "XR CHEST PORTABLE",
        "XR HAND LEFT 3 VIEWS",
        "XR HAND RIGHT 3 VIEWS",
        "XR SPINE CERVICAL 3 VIEWS",
        "XR SPINE LUMBAR 3 VIEWS",
        "XR PELVIS AP",
        "XR KNEE LEFT 2 VIEWS",
        "XR ANKLE RIGHT 3 VIEWS"
    ],
    "US": [
        "US ABDOMEN COMPLETE",
        "US ABDOMEN LIMITED",
        "US PELVIS TRANSABDOMINAL",
        "US THYROID WITH DOPPLER",
        "US RENAL WITH DOPPLER",
        "US CAROTID DUPLEX BILATERAL",
        "US ECHOCARDIOGRAM COMPLETE",
        "US VENOUS LOWER EXTREMITY LEFT",
        "US OBSTETRIC SECOND TRIMESTER"
    ]
}

# Equipment manufacturers and models
EQUIPMENT = {
    "CT": [
        {"manufacturer": "GE Healthcare", "model": "Revolution CT", "station": "CT-GE-01"},
        {"manufacturer": "Siemens Healthineers", "model": "SOMATOM Force", "station": "CT-SIE-01"},
        {"manufacturer": "Philips", "model": "Spectral CT 7500", "station": "CT-PHI-01"},
        {"manufacturer": "Canon Medical", "model": "Aquilion ONE", "station": "CT-CAN-01"}
    ],
    "MR": [
        {"manufacturer": "GE Healthcare", "model": "SIGNA Premier", "station": "MR-GE-01"},
        {"manufacturer": "Siemens Healthineers", "model": "MAGNETOM Vida", "station": "MR-SIE-01"},
        {"manufacturer": "Philips", "model": "Ingenia Ambition", "station": "MR-PHI-01"}
    ],
    "XR": [
        {"manufacturer": "GE Healthcare", "model": "Definium 656", "station": "XR-GE-01"},
        {"manufacturer": "Siemens Healthineers", "model": "Ysio Max", "station": "XR-SIE-01"},
        {"manufacturer": "Fujifilm", "model": "FDR D-EVO II", "station": "XR-FUJ-01"}
    ],
    "US": [
        {"manufacturer": "GE Healthcare", "model": "LOGIQ E10", "station": "US-GE-01"},
        {"manufacturer": "Philips", "model": "EPIQ Elite", "station": "US-PHI-01"},
        {"manufacturer": "Siemens Healthineers", "model": "ACUSON Sequoia", "station": "US-SIE-01"}
    ]
}

# COMMAND ----------

def generate_synthetic_pixel_data(modality: str, rows: int, cols: int, instance_number: int = 1) -> bytes:
    """
    Generate synthetic pixel data that resembles real medical images.
    Uses np.indices for proper 2D coordinate arrays.
    """
    np.random.seed(hash(f"{modality}-{instance_number}") % (2**32))
    
    # Create 2D coordinate grids (y is row index, x is column index)
    y, x = np.indices((rows, cols))
    
    if modality == "CT":
        # CT images: Hounsfield units, typically -1000 (air) to +3000 (dense bone)
        base = np.random.normal(40, 15, (rows, cols)).astype(np.float32)
        
        # Add body outline (ellipse)
        cy, cx = rows // 2, cols // 2
        body_mask = ((x - cx) / (cols * 0.4))**2 + ((y - cy) / (rows * 0.35))**2 <= 1
        base[body_mask] += 40
        
        # Add organ-like structures
        for _ in range(np.random.randint(3, 7)):
            ox = np.random.randint(cols // 4, 3 * cols // 4)
            oy = np.random.randint(rows // 4, 3 * rows // 4)
            radius = np.random.randint(15, 50)
            organ_mask = (x - ox)**2 + (y - oy)**2 <= radius**2
            base[organ_mask] += np.random.randint(20, 100)
        
        # Add spine (bright posterior structure)
        spine_mask = ((x - cx) / 20)**2 + ((y - (rows * 0.7)) / 15)**2 <= 1
        base[spine_mask] = np.random.randint(200, 400)
        
        # Convert to stored values (offset by 1024 for CT)
        pixel_data = (base + 1024).clip(0, 4095).astype(np.uint16)
        
    elif modality == "MR":
        # MRI: T1/T2 weighted appearance
        base = np.random.normal(300, 50, (rows, cols)).astype(np.float32)
        
        # Brain-like structure for head MRI
        cy, cx = rows // 2, cols // 2
        
        # Skull (dark ring)
        skull_outer = ((x - cx) / (cols * 0.42))**2 + ((y - cy) / (rows * 0.45))**2 <= 1
        skull_inner = ((x - cx) / (cols * 0.38))**2 + ((y - cy) / (rows * 0.41))**2 <= 1
        skull_mask = skull_outer & ~skull_inner
        base[skull_mask] = np.random.randint(100, 200)
        
        # Brain tissue
        brain_mask = skull_inner
        num_brain_pixels = np.sum(brain_mask)
        if num_brain_pixels > 0:
            base[brain_mask] = np.random.normal(600, 80, size=num_brain_pixels)
        
        # Ventricles (CSF - bright on T2)
        vent_mask = ((x - cx) / 25)**2 + ((y - cy) / 15)**2 <= 1
        base[vent_mask] = np.random.randint(800, 1000)
        
        pixel_data = base.clip(0, 4095).astype(np.uint16)
        
    elif modality in ["XR", "CR"]:
        # X-ray: high values = more penetration (brighter)
        base = np.ones((rows, cols), dtype=np.float32) * 3000
        
        # Simulate chest X-ray
        cy, cx = rows // 2, cols // 2
        
        # Lung fields (darker = more air)
        lung_left = ((x - (cx - cols * 0.15)) / (cols * 0.2))**2 + ((y - cy) / (rows * 0.35))**2 <= 1
        lung_right = ((x - (cx + cols * 0.15)) / (cols * 0.2))**2 + ((y - cy) / (rows * 0.35))**2 <= 1
        
        num_left = np.sum(lung_left)
        num_right = np.sum(lung_right)
        if num_left > 0:
            base[lung_left] = np.random.normal(2500, 200, size=num_left)
        if num_right > 0:
            base[lung_right] = np.random.normal(2500, 200, size=num_right)
        
        # Heart shadow (mediastinum)
        heart = ((x - (cx + cols * 0.05)) / (cols * 0.15))**2 + ((y - (cy + rows * 0.1)) / (rows * 0.2))**2 <= 1
        num_heart = np.sum(heart)
        if num_heart > 0:
            base[heart] = np.random.normal(1500, 100, size=num_heart)
        
        # Ribs (curved lines)
        for i in range(6):
            rib_y = int(cy - rows * 0.25 + i * rows * 0.1)
            for side in [-1, 1]:
                for rx in range(int(cols * 0.1), int(cols * 0.4)):
                    ry = rib_y + int(np.sin(rx / 50) * 10)
                    px = cx + side * rx
                    if 0 <= px < cols and 0 <= ry < rows:
                        y_start, y_end = max(0, ry-2), min(rows, ry+2)
                        x_start, x_end = max(0, px-1), min(cols, px+1)
                        base[y_start:y_end, x_start:x_end] = np.random.randint(800, 1200)
        
        # Spine (central vertical structure)
        spine_mask = np.abs(x - cx) < cols * 0.03
        num_spine = np.sum(spine_mask)
        if num_spine > 0:
            base[spine_mask] = np.random.normal(1000, 100, size=num_spine)
        
        pixel_data = base.clip(0, 4095).astype(np.uint16)
        
    else:  # US - Ultrasound
        # Ultrasound: characteristic speckle pattern with fan shape
        base = np.random.exponential(400, (rows, cols)).astype(np.float32)
        
        # Fan/sector shape
        cx = cols // 2
        for row in range(rows):
            depth_factor = row / rows
            fan_half_width = int(cols * 0.15 + cols * 0.35 * depth_factor)
            x_start = max(0, cx - fan_half_width)
            x_end = min(cols, cx + fan_half_width)
            
            # Outside fan is black
            base[row, :x_start] = 0
            base[row, x_end:] = 0
            
            # Add depth-dependent attenuation
            base[row, x_start:x_end] *= (1 - depth_factor * 0.4)
        
        # Add some hyperechoic structures
        for _ in range(np.random.randint(2, 5)):
            sy = np.random.randint(rows // 4, 3 * rows // 4)
            sx = np.random.randint(cols // 3, 2 * cols // 3)
            sr = np.random.randint(10, 30)
            structure = (x - sx)**2 + (y - sy)**2 <= sr**2
            base[structure] = np.random.randint(600, 900)
        
        pixel_data = base.clip(0, 4095).astype(np.uint16)
    
    return pixel_data.tobytes()

# COMMAND ----------

def create_dicom_file(
    patient_id: str,
    patient_name: str,
    patient_birth_date: str,
    patient_sex: str,
    modality: str,
    study_uid: str,
    study_date: str,
    study_time: str,
    study_description: str,
    series_uid: str,
    series_number: int,
    series_description: str,
    instance_number: int,
    equipment: dict,
    image_size: tuple = (512, 512)
) -> FileDataset:
    """
    Create a DICOM file with complete and realistic metadata.
    """
    # Determine SOP Class based on modality
    sop_class_uid = SOP_CLASS_UIDS.get(modality, pydicom.uid.SecondaryCaptureImageStorage)
    
    # Create file meta information
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = sop_class_uid
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.ImplementationVersionName = "LAB_FUTURE_1.0"
    
    # Create the FileDataset
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    # ----- Patient Module -----
    ds.PatientID = patient_id
    ds.PatientName = patient_name
    ds.PatientBirthDate = patient_birth_date
    ds.PatientSex = patient_sex
    ds.PatientAge = f"{(datetime.now() - datetime.strptime(patient_birth_date, '%Y%m%d')).days // 365:03d}Y"
    
    # ----- Study Module -----
    ds.StudyInstanceUID = study_uid
    ds.StudyDate = study_date
    ds.StudyTime = study_time
    ds.StudyDescription = study_description
    ds.AccessionNumber = f"ACC{np.random.randint(1000000, 9999999)}"
    ds.ReferringPhysicianName = f"Dr. {np.random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones'])}"
    ds.StudyID = f"STUDY{np.random.randint(10000, 99999)}"
    
    # ----- Series Module -----
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = series_number
    ds.SeriesDescription = series_description
    ds.Modality = modality
    ds.SeriesDate = study_date
    ds.SeriesTime = study_time
    ds.BodyPartExamined = _get_body_part(study_description)
    ds.PatientPosition = "HFS" if modality in ["CT", "MR"] else ""
    
    # ----- Equipment Module -----
    ds.Manufacturer = equipment["manufacturer"]
    ds.ManufacturerModelName = equipment["model"]
    ds.StationName = equipment["station"]
    ds.InstitutionName = "Lab of the Future Medical Center"
    ds.InstitutionAddress = "123 Innovation Drive, Future City, FC 12345"
    ds.SoftwareVersions = "1.0.0"
    ds.DeviceSerialNumber = f"SN{np.random.randint(100000, 999999)}"
    
    # ----- Instance Module -----
    ds.SOPClassUID = sop_class_uid
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.InstanceNumber = instance_number
    ds.ContentDate = study_date
    ds.ContentTime = study_time
    ds.InstanceCreationDate = study_date
    ds.InstanceCreationTime = study_time
    
    # ----- Image Module -----
    rows, cols = image_size
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PlanarConfiguration = 0
    
    # Generate pixel data
    ds.PixelData = generate_synthetic_pixel_data(modality, rows, cols, instance_number)
    
    # ----- Spatial Information -----
    ds.ImagePositionPatient = [0.0, 0.0, float(instance_number * 2.5)]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.PixelSpacing = [0.5, 0.5]
    ds.SliceThickness = 2.5
    ds.SliceLocation = float(instance_number * 2.5)
    
    # ----- Modality-specific attributes -----
    if modality == "CT":
        ds.KVP = np.random.choice([80, 100, 120, 140])
        ds.XRayTubeCurrent = np.random.randint(100, 400)
        ds.ConvolutionKernel = np.random.choice(["STANDARD", "BONE", "LUNG", "SOFT"])
        ds.RescaleIntercept = -1024
        ds.RescaleSlope = 1
        ds.WindowCenter = 40
        ds.WindowWidth = 400
        
    elif modality == "MR":
        ds.MagneticFieldStrength = np.random.choice([1.5, 3.0])
        ds.EchoTime = np.random.uniform(10, 100)
        ds.RepetitionTime = np.random.uniform(500, 3000)
        ds.FlipAngle = np.random.choice([15, 30, 45, 60, 90])
        ds.ImagingFrequency = 63.87 if ds.MagneticFieldStrength == 1.5 else 127.74
        ds.SequenceName = np.random.choice(["SE", "GRE", "FSE", "EPI"])
        
    elif modality in ["XR", "CR"]:
        ds.KVP = np.random.choice([60, 70, 80, 90, 100, 110, 120])
        ds.ExposureTime = np.random.randint(1, 100)
        ds.XRayTubeCurrent = np.random.randint(50, 300)
        ds.DistanceSourceToDetector = 1000.0
        ds.DistanceSourceToPatient = 900.0
        ds.WindowCenter = 2048
        ds.WindowWidth = 4096
        
    elif modality == "US":
        ds.TransducerType = np.random.choice(["CURVED ARRAY", "LINEAR", "PHASED ARRAY", "SECTOR"])
        ds.TransducerFrequency = np.random.choice([2.5, 3.5, 5.0, 7.5, 10.0])
        ds.MechanicalIndex = round(np.random.uniform(0.5, 1.2), 2)
        ds.ThermalIndex = round(np.random.uniform(0.3, 1.0), 2)
    
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    
    return ds


def _get_body_part(study_description: str) -> str:
    """Extract body part from study description."""
    desc_upper = study_description.upper()
    body_parts = {
        "HEAD": "HEAD", "BRAIN": "HEAD", "CHEST": "CHEST", "THORAX": "CHEST",
        "ABDOMEN": "ABDOMEN", "PELVIS": "PELVIS", "SPINE": "SPINE",
        "CERVICAL": "CSPINE", "LUMBAR": "LSPINE", "KNEE": "KNEE",
        "SHOULDER": "SHOULDER", "HAND": "HAND", "ANKLE": "ANKLE",
        "CARDIAC": "HEART", "THYROID": "THYROID", "RENAL": "KIDNEY",
        "CAROTID": "NECK", "OBSTETRIC": "PELVIS"
    }
    for key, part in body_parts.items():
        if key in desc_upper:
            return part
    return "UNKNOWN"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Sample DICOM Dataset
# MAGIC 
# MAGIC Generate a realistic set of DICOM files representing multiple patients, studies, and imaging modalities.

# COMMAND ----------

# Configuration for data generation
NUM_PATIENTS = 25
STUDIES_PER_PATIENT = (1, 4)  # min, max studies per patient
SLICES_CT_MR = (5, 15)  # min, max slices for volumetric imaging
SLICES_2D = (1, 3)  # min, max for 2D modalities (XR, US)

print(f"Generating synthetic DICOM dataset...")
print(f"  Patients: {NUM_PATIENTS}")
print(f"  Studies per patient: {STUDIES_PER_PATIENT[0]}-{STUDIES_PER_PATIENT[1]}")
print(f"  Modalities: CT, MR, XR, US")

# COMMAND ----------

# Generate the dataset
generated_files = []
modalities = ["CT", "MR", "XR", "US"]

for patient_num in range(NUM_PATIENTS):
    # Generate patient demographics
    patient_id = f"PT{100000 + patient_num}"
    first_names = ["John", "Jane", "Michael", "Sarah", "Robert", "Emily", "David", "Lisa", "James", "Maria"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
    patient_name = f"{np.random.choice(last_names)}^{np.random.choice(first_names)}"
    
    # Random birth date (age 20-85)
    age_days = np.random.randint(20 * 365, 85 * 365)
    birth_date = (datetime.now() - timedelta(days=age_days)).strftime("%Y%m%d")
    patient_sex = np.random.choice(["M", "F"])
    
    # Generate studies for this patient
    num_studies = np.random.randint(STUDIES_PER_PATIENT[0], STUDIES_PER_PATIENT[1] + 1)
    
    for study_num in range(num_studies):
        # Select modality and study description
        modality = np.random.choice(modalities)
        study_description = np.random.choice(STUDY_DESCRIPTIONS[modality])
        equipment = np.random.choice(EQUIPMENT[modality])
        
        # Generate study UIDs and timing
        study_uid = generate_uid()
        study_date = (datetime.now() - timedelta(days=np.random.randint(0, 365))).strftime("%Y%m%d")
        study_time = f"{np.random.randint(6, 22):02d}{np.random.randint(0, 60):02d}{np.random.randint(0, 60):02d}"
        
        # Generate series
        series_uid = generate_uid()
        series_number = 1
        series_description = study_description
        
        # Determine number of instances
        if modality in ["CT", "MR"]:
            num_instances = np.random.randint(SLICES_CT_MR[0], SLICES_CT_MR[1] + 1)
        else:
            num_instances = np.random.randint(SLICES_2D[0], SLICES_2D[1] + 1)
        
        # Generate instances
        for instance_num in range(1, num_instances + 1):
            ds = create_dicom_file(
                patient_id=patient_id,
                patient_name=patient_name,
                patient_birth_date=birth_date,
                patient_sex=patient_sex,
                modality=modality,
                study_uid=study_uid,
                study_date=study_date,
                study_time=study_time,
                study_description=study_description,
                series_uid=series_uid,
                series_number=series_number,
                series_description=series_description,
                instance_number=instance_num,
                equipment=equipment
            )
            
            generated_files.append({
                "dataset": ds,
                "patient_id": patient_id,
                "patient_name": patient_name,
                "patient_sex": patient_sex,
                "modality": modality,
                "study_uid": study_uid,
                "study_date": study_date,
                "study_description": study_description,
                "series_uid": series_uid,
                "sop_uid": ds.SOPInstanceUID,
                "instance_number": instance_num,
                "manufacturer": equipment["manufacturer"]
            })

print(f"\nGenerated {len(generated_files)} DICOM instances")

# Show distribution
from collections import Counter
modality_counts = Counter([f["modality"] for f in generated_files])
print(f"\nModality distribution:")
for mod, count in sorted(modality_counts.items()):
    print(f"  {mod}: {count} instances")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save DICOM Files to Unity Catalog Volume
# MAGIC 
# MAGIC Save files following the standard DICOM hierarchy:
# MAGIC `<Volume>/<PatientID>/<StudyUID>/<SeriesUID>/<SOPInstanceUID>.dcm`

# COMMAND ----------

import os

saved_count = 0
save_errors = []

for file_info in generated_files:
    ds = file_info["dataset"]
    patient_id = file_info["patient_id"]
    study_uid = file_info["study_uid"][:16]  # Truncate for filesystem compatibility
    series_uid = file_info["series_uid"][:16]
    sop_uid = file_info["sop_uid"][:16]
    
    # Create directory hierarchy using Unity Catalog Volume path
    # UC Volumes use /Volumes/ path directly (not /dbfs/)
    dir_path = f"{DICOM_VOLUME}/{patient_id}/{study_uid}/{series_uid}"
    
    try:
        os.makedirs(dir_path, exist_ok=True)
        
        # Save DICOM file
        file_path = f"{dir_path}/{sop_uid}.dcm"
        ds.save_as(file_path)
        
        # Update file_info with saved path
        file_info["file_path"] = f"{DICOM_VOLUME}/{patient_id}/{study_uid}/{series_uid}/{sop_uid}.dcm"
        saved_count += 1
        
    except Exception as e:
        save_errors.append(f"{patient_id}: {str(e)}")

print(f"Saved {saved_count} DICOM files to {DICOM_VOLUME}")
if save_errors:
    print(f"Errors: {len(save_errors)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Metadata Catalog Table
# MAGIC 
# MAGIC Store metadata for all generated files in a Delta table for easy querying.

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType
from pyspark.sql.functions import current_timestamp, to_date, to_timestamp

# Define explicit schema for the metadata table
metadata_schema = StructType([
    StructField("file_path", StringType(), True),
    StructField("patient_id", StringType(), True),
    StructField("patient_name", StringType(), True),
    StructField("patient_sex", StringType(), True),
    StructField("modality", StringType(), True),
    StructField("study_instance_uid", StringType(), True),
    StructField("study_date", StringType(), True),
    StructField("study_description", StringType(), True),
    StructField("series_instance_uid", StringType(), True),
    StructField("sop_instance_uid", StringType(), True),
    StructField("instance_number", IntegerType(), True),
    StructField("manufacturer", StringType(), True)
])

# Create metadata records
metadata_records = []
for f in generated_files:
    if "file_path" in f:  # Only include successfully saved files
        metadata_records.append({
            "file_path": str(f["file_path"]),
            "patient_id": str(f["patient_id"]),
            "patient_name": str(f["patient_name"]),
            "patient_sex": str(f["patient_sex"]),
            "modality": str(f["modality"]),
            "study_instance_uid": str(f["study_uid"]),
            "study_date": str(f["study_date"]),
            "study_description": str(f["study_description"]),
            "series_instance_uid": str(f["series_uid"]),
            "sop_instance_uid": str(f["sop_uid"]),
            "instance_number": int(f["instance_number"]),
            "manufacturer": str(f["manufacturer"])
        })

# Create DataFrame with explicit schema
metadata_df = spark.createDataFrame(metadata_records, schema=metadata_schema)

# Add ingestion timestamp
metadata_df = metadata_df.withColumn("ingestion_timestamp", current_timestamp())

# Save to Delta table
metadata_df.write.mode("overwrite").saveAsTable(DICOM_METADATA_TABLE)

print(f"Created metadata table: {DICOM_METADATA_TABLE}")
print(f"Total records: {metadata_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Generated Data

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Overview of generated DICOM data
# MAGIC SELECT 
# MAGIC   modality,
# MAGIC   COUNT(*) as instance_count,
# MAGIC   COUNT(DISTINCT patient_id) as patients,
# MAGIC   COUNT(DISTINCT study_instance_uid) as studies,
# MAGIC   COUNT(DISTINCT series_instance_uid) as series
# MAGIC FROM lab_of_the_future.healthcare_data.dicom_metadata
# MAGIC GROUP BY modality
# MAGIC ORDER BY instance_count DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Sample of generated files
# MAGIC SELECT 
# MAGIC   patient_id,
# MAGIC   patient_name,
# MAGIC   modality,
# MAGIC   study_description,
# MAGIC   instance_number,
# MAGIC   manufacturer,
# MAGIC   file_path
# MAGIC FROM lab_of_the_future.healthcare_data.dicom_metadata
# MAGIC ORDER BY patient_id, study_instance_uid, instance_number
# MAGIC LIMIT 20;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Study distribution by manufacturer
# MAGIC SELECT 
# MAGIC   manufacturer,
# MAGIC   modality,
# MAGIC   COUNT(DISTINCT study_instance_uid) as study_count,
# MAGIC   COUNT(*) as instance_count
# MAGIC FROM lab_of_the_future.healthcare_data.dicom_metadata
# MAGIC GROUP BY manufacturer, modality
# MAGIC ORDER BY manufacturer, modality;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC This notebook has:
# MAGIC 1. Created Unity Catalog volumes for DICOM storage
# MAGIC 2. Generated synthetic DICOM files with realistic medical imaging metadata
# MAGIC 3. Saved files to UC Volume with proper DICOM hierarchy
# MAGIC 4. Created a metadata catalog table for downstream processing
# MAGIC 
# MAGIC ### Output Artifacts
# MAGIC - **Volume**: `/Volumes/lab_of_the_future/healthcare_data/dicom_raw/`
# MAGIC - **Table**: `lab_of_the_future.healthcare_data.dicom_metadata`
# MAGIC 
# MAGIC ### Next Steps
# MAGIC - **Notebook 2b**: Process DICOM files with the Databricks Pixels Solution Accelerator
# MAGIC - **Notebook 3**: Combine with EKG data in the Surgery Room Digital Twin
