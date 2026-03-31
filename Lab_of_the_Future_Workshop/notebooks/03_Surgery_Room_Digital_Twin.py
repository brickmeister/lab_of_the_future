# Databricks notebook source
# MAGIC %md
# MAGIC # Lab of the Future Workshop - Notebook 3
# MAGIC ## Surgery Room Digital Twin Prototype
# MAGIC 
# MAGIC This notebook demonstrates how to build a **digital twin** of a surgery room by combining:
# MAGIC - **EKG data** from Notebook 1 (patient vital signs)
# MAGIC - **DICOM imaging data** from Notebook 2 (medical images)
# MAGIC - **Video data** from operating room cameras (procedure monitoring)
# MAGIC 
# MAGIC ### What is a Digital Twin?
# MAGIC A digital twin is a virtual representation of a physical environment that integrates real-time 
# MAGIC and historical data to provide insights, enable simulation, and support decision-making.
# MAGIC 
# MAGIC ### Learning Objectives
# MAGIC - Integrate multi-modal healthcare data (vitals, images, video)
# MAGIC - Build real-time monitoring dashboards
# MAGIC - Implement alerting for critical events
# MAGIC - Create a unified view of surgery room operations
# MAGIC 
# MAGIC ### Architecture
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────┐
# MAGIC │                    Surgery Room Digital Twin                     │
# MAGIC ├─────────────────────────────────────────────────────────────────┤
# MAGIC │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
# MAGIC │  │  EKG Data   │  │ DICOM Data  │  │     Video Streams      │  │
# MAGIC │  │  (Vitals)   │  │  (Images)   │  │  (Room Monitoring)     │  │
# MAGIC │  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
# MAGIC │         │                │                     │                │
# MAGIC │         └────────────────┼─────────────────────┘                │
# MAGIC │                          ▼                                      │
# MAGIC │              ┌───────────────────────┐                          │
# MAGIC │              │   Unified Data Lake   │                          │
# MAGIC │              │   (Unity Catalog)     │                          │
# MAGIC │              └───────────┬───────────┘                          │
# MAGIC │                          ▼                                      │
# MAGIC │  ┌─────────────────────────────────────────────────────────┐   │
# MAGIC │  │              Digital Twin Engine                         │   │
# MAGIC │  │  • Real-time State Computation                          │   │
# MAGIC │  │  • Anomaly Detection                                    │   │
# MAGIC │  │  • Predictive Analytics                                 │   │
# MAGIC │  └─────────────────────────────────────────────────────────┘   │
# MAGIC └─────────────────────────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies

# COMMAND ----------

# MAGIC %pip install plotly pandas numpy opencv-python-headless pillow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

CATALOG = "lab_of_the_future"
SCHEMA = "healthcare_data"
DIGITAL_TWIN_SCHEMA = "digital_twin"

# Source tables from previous notebooks
EKG_TABLE = f"{CATALOG}.{SCHEMA}.ekg_silver"
DICOM_FEATURES_TABLE = f"{CATALOG}.{SCHEMA}.dicom_features"
DICOM_METADATA_TABLE = f"{CATALOG}.{SCHEMA}.dicom_metadata"

# Digital twin tables
SURGERY_SESSIONS_TABLE = f"{CATALOG}.{DIGITAL_TWIN_SCHEMA}.surgery_sessions"
ROOM_STATE_TABLE = f"{CATALOG}.{DIGITAL_TWIN_SCHEMA}.room_state"
VIDEO_METADATA_TABLE = f"{CATALOG}.{DIGITAL_TWIN_SCHEMA}.video_metadata"
ALERTS_TABLE = f"{CATALOG}.{DIGITAL_TWIN_SCHEMA}.alerts"

print(f"Digital Twin Schema: {CATALOG}.{DIGITAL_TWIN_SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup: Create Digital Twin Schema

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create schema for digital twin data
# MAGIC CREATE SCHEMA IF NOT EXISTS lab_of_the_future.digital_twin
# MAGIC COMMENT 'Surgery Room Digital Twin - integrated real-time monitoring data';
# MAGIC 
# MAGIC -- Create volume for video data
# MAGIC CREATE VOLUME IF NOT EXISTS lab_of_the_future.digital_twin.video_recordings
# MAGIC COMMENT 'Operating room video recordings and frame extracts';

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Surgery Room Model
# MAGIC 
# MAGIC Create the data model for representing surgery rooms and their digital twin state

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *
from datetime import datetime, timedelta
import numpy as np
import json

# Surgery Room schema
surgery_room_schema = StructType([
    StructField("room_id", StringType(), False),
    StructField("room_name", StringType(), False),
    StructField("floor", IntegerType(), False),
    StructField("building", StringType(), False),
    StructField("room_type", StringType(), False),  # OR, ICU, Recovery, etc.
    StructField("equipment", ArrayType(StringType()), True),
    StructField("capacity", IntegerType(), True),
    StructField("created_at", TimestampType(), False)
])

# Surgery Session schema
surgery_session_schema = StructType([
    StructField("session_id", StringType(), False),
    StructField("room_id", StringType(), False),
    StructField("patient_id", StringType(), False),
    StructField("procedure_type", StringType(), False),
    StructField("lead_surgeon", StringType(), False),
    StructField("surgical_team", ArrayType(StringType()), True),
    StructField("scheduled_start", TimestampType(), False),
    StructField("scheduled_end", TimestampType(), False),
    StructField("actual_start", TimestampType(), True),
    StructField("actual_end", TimestampType(), True),
    StructField("status", StringType(), False),  # scheduled, in_progress, completed, cancelled
    StructField("created_at", TimestampType(), False)
])

# Room State schema (digital twin state)
room_state_schema = StructType([
    StructField("state_id", StringType(), False),
    StructField("room_id", StringType(), False),
    StructField("session_id", StringType(), True),
    StructField("timestamp", TimestampType(), False),
    StructField("patient_heart_rate", IntegerType(), True),
    StructField("patient_heart_rate_variability", DoubleType(), True),
    StructField("patient_status", StringType(), True),  # stable, warning, critical
    StructField("active_imaging_modality", StringType(), True),
    StructField("images_taken", IntegerType(), True),
    StructField("video_feed_active", BooleanType(), True),
    StructField("personnel_count", IntegerType(), True),
    StructField("room_temperature_c", DoubleType(), True),
    StructField("room_humidity_pct", DoubleType(), True),
    StructField("equipment_status", MapType(StringType(), StringType()), True),
    StructField("alerts_active", ArrayType(StringType()), True)
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Sample Surgery Room Data

# COMMAND ----------

# Define surgery rooms
surgery_rooms = [
    {
        "room_id": "OR-001",
        "room_name": "Operating Room 1 - Cardiac",
        "floor": 3,
        "building": "Main Hospital",
        "room_type": "OR",
        "equipment": ["EKG Monitor", "Anesthesia Machine", "Surgical Robot", "C-Arm Fluoroscopy", "Ultrasound"],
        "capacity": 8,
        "created_at": datetime.now()
    },
    {
        "room_id": "OR-002",
        "room_name": "Operating Room 2 - Neuro",
        "floor": 3,
        "building": "Main Hospital",
        "room_type": "OR",
        "equipment": ["EKG Monitor", "Anesthesia Machine", "Neuro Navigation", "MRI Compatible Tools", "Microscope"],
        "capacity": 6,
        "created_at": datetime.now()
    },
    {
        "room_id": "OR-003",
        "room_name": "Operating Room 3 - General",
        "floor": 2,
        "building": "Main Hospital",
        "room_type": "OR",
        "equipment": ["EKG Monitor", "Anesthesia Machine", "Laparoscopic Tower", "X-Ray"],
        "capacity": 6,
        "created_at": datetime.now()
    },
    {
        "room_id": "ICU-001",
        "room_name": "Intensive Care Unit - Bay 1",
        "floor": 4,
        "building": "Critical Care Wing",
        "room_type": "ICU",
        "equipment": ["Multi-Parameter Monitor", "Ventilator", "Infusion Pumps", "Portable Ultrasound"],
        "capacity": 4,
        "created_at": datetime.now()
    }
]

# Create surgery rooms DataFrame
rooms_df = spark.createDataFrame(surgery_rooms, surgery_room_schema)
rooms_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{DIGITAL_TWIN_SCHEMA}.surgery_rooms")

display(rooms_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Surgery Sessions

# COMMAND ----------

import random

# Procedure types by room
procedure_types = {
    "OR-001": ["CABG", "Valve Replacement", "Heart Transplant", "Angioplasty"],
    "OR-002": ["Craniotomy", "Spinal Fusion", "Deep Brain Stimulation", "Tumor Resection"],
    "OR-003": ["Appendectomy", "Cholecystectomy", "Hernia Repair", "Bowel Resection"],
    "ICU-001": ["Post-Op Monitoring", "Critical Care", "Ventilator Management"]
}

surgeons = [f"Dr. {name}" for name in ["Johnson", "Williams", "Chen", "Patel", "Rodriguez", "Kim", "Ahmed", "Smith"]]
nurses = [f"Nurse {name}" for name in ["Taylor", "Brown", "Davis", "Wilson", "Martinez", "Anderson", "Thomas", "Garcia"]]
anesthesiologists = [f"Dr. {name} (Anesthesia)" for name in ["Lee", "Clark", "Lewis", "Hall", "Young"]]

# Generate surgery sessions
base_time = datetime.now()
surgery_sessions = []

for i in range(30):  # 30 surgery sessions
    room_id = random.choice(list(procedure_types.keys()))
    procedure = random.choice(procedure_types[room_id])
    
    # Schedule surgeries across the past 7 days
    scheduled_start = base_time - timedelta(
        days=random.randint(0, 7),
        hours=random.randint(6, 18),
        minutes=random.choice([0, 30])
    )
    duration_hours = random.uniform(1, 6)
    scheduled_end = scheduled_start + timedelta(hours=duration_hours)
    
    # Determine status based on time
    if scheduled_start > base_time:
        status = "scheduled"
        actual_start = None
        actual_end = None
    elif scheduled_end < base_time:
        status = "completed"
        actual_start = scheduled_start + timedelta(minutes=random.randint(-15, 30))
        actual_end = actual_start + timedelta(hours=duration_hours * random.uniform(0.9, 1.3))
    else:
        status = "in_progress"
        actual_start = scheduled_start + timedelta(minutes=random.randint(-15, 15))
        actual_end = None
    
    # Build surgical team
    team = [
        random.choice(anesthesiologists),
        random.choice(nurses),
        random.choice(nurses)
    ]
    
    session = {
        "session_id": f"SURG-{100000 + i}",
        "room_id": room_id,
        "patient_id": f"PT-{10000 + random.randint(0, 19)}",
        "procedure_type": procedure,
        "lead_surgeon": random.choice(surgeons),
        "surgical_team": team,
        "scheduled_start": scheduled_start,
        "scheduled_end": scheduled_end,
        "actual_start": actual_start,
        "actual_end": actual_end,
        "status": status,
        "created_at": scheduled_start - timedelta(days=random.randint(1, 14))
    }
    surgery_sessions.append(session)

# Create sessions DataFrame
sessions_df = spark.createDataFrame(surgery_sessions, surgery_session_schema)
sessions_df.write.mode("overwrite").saveAsTable(SURGERY_SESSIONS_TABLE)

print(f"Created {len(surgery_sessions)} surgery sessions")
display(sessions_df.orderBy("scheduled_start", ascending=False).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Room State Time Series (Digital Twin State)

# COMMAND ----------

def generate_room_state(room_id, session_id, timestamp, patient_status_base="stable"):
    """
    Generate a snapshot of the digital twin room state.
    """
    # Patient vitals with some variation based on status
    if patient_status_base == "critical":
        hr_base = random.randint(110, 160)
        hrv_base = random.uniform(5, 15)
    elif patient_status_base == "warning":
        hr_base = random.randint(90, 120)
        hrv_base = random.uniform(15, 30)
    else:
        hr_base = random.randint(60, 90)
        hrv_base = random.uniform(30, 60)
    
    # Add time-based variation
    hr = hr_base + random.randint(-5, 5)
    hrv = hrv_base + random.uniform(-5, 5)
    
    # Determine patient status from vitals
    if hr > 130 or hr < 45:
        patient_status = "critical"
    elif hr > 110 or hr < 55:
        patient_status = "warning"
    else:
        patient_status = "stable"
    
    # Equipment status
    equipment_status = {
        "EKG Monitor": random.choice(["online", "online", "online", "calibrating"]),
        "Anesthesia Machine": random.choice(["active", "active", "standby"]),
        "Ventilator": random.choice(["active", "active", "standby"]),
        "Surgical Lights": "on",
        "Suction": random.choice(["active", "standby"])
    }
    
    # Active alerts
    alerts = []
    if patient_status == "critical":
        alerts.append("CRITICAL: Heart rate anomaly detected")
    if patient_status == "warning":
        alerts.append("WARNING: Elevated heart rate")
    if random.random() < 0.05:
        alerts.append("INFO: Equipment calibration due")
    
    return {
        "state_id": f"STATE-{random.randint(1000000, 9999999)}",
        "room_id": room_id,
        "session_id": session_id,
        "timestamp": timestamp,
        "patient_heart_rate": hr,
        "patient_heart_rate_variability": round(hrv, 2),
        "patient_status": patient_status,
        "active_imaging_modality": random.choice([None, "X-Ray", "Ultrasound", "Fluoroscopy"]),
        "images_taken": random.randint(0, 20),
        "video_feed_active": random.choice([True, True, True, False]),
        "personnel_count": random.randint(3, 8),
        "room_temperature_c": round(random.uniform(18, 22), 1),
        "room_humidity_pct": round(random.uniform(40, 60), 1),
        "equipment_status": equipment_status,
        "alerts_active": alerts
    }

# Generate room state history
room_states = []
base_time = datetime.now()

# For each completed or in-progress session, generate state snapshots
for session in surgery_sessions:
    if session["status"] in ["completed", "in_progress"]:
        start_time = session["actual_start"]
        end_time = session["actual_end"] if session["actual_end"] else base_time
        
        # Generate state every 5 minutes
        current_time = start_time
        while current_time <= end_time:
            # Vary patient status during surgery
            elapsed_pct = (current_time - start_time).total_seconds() / max(1, (end_time - start_time).total_seconds())
            
            # Simulate typical surgery phases
            if elapsed_pct < 0.1:  # Induction
                status_base = "warning"
            elif elapsed_pct < 0.8:  # Main procedure
                status_base = random.choices(["stable", "warning", "critical"], weights=[85, 12, 3])[0]
            else:  # Recovery phase
                status_base = "stable"
            
            state = generate_room_state(
                session["room_id"],
                session["session_id"],
                current_time,
                status_base
            )
            room_states.append(state)
            
            current_time += timedelta(minutes=5)

# Create room state DataFrame
states_df = spark.createDataFrame(room_states, room_state_schema)
states_df.write.mode("overwrite").saveAsTable(ROOM_STATE_TABLE)

print(f"Generated {len(room_states)} room state snapshots")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Video Metadata
# MAGIC 
# MAGIC Track video recordings from operating room cameras

# COMMAND ----------

video_metadata_schema = StructType([
    StructField("video_id", StringType(), False),
    StructField("room_id", StringType(), False),
    StructField("session_id", StringType(), True),
    StructField("camera_id", StringType(), False),
    StructField("camera_position", StringType(), False),
    StructField("recording_start", TimestampType(), False),
    StructField("recording_end", TimestampType(), True),
    StructField("duration_seconds", IntegerType(), True),
    StructField("resolution", StringType(), False),
    StructField("fps", IntegerType(), False),
    StructField("file_path", StringType(), True),
    StructField("file_size_mb", DoubleType(), True),
    StructField("status", StringType(), False),
    StructField("ai_analysis_complete", BooleanType(), False),
    StructField("detected_events", ArrayType(StringType()), True)
])

# Camera positions in OR
camera_positions = ["Overhead Main", "Surgical Field", "Anesthesia Station", "Entry Door", "Equipment Bay"]

# Generate video metadata
video_records = []
for session in surgery_sessions:
    if session["status"] in ["completed", "in_progress"]:
        start_time = session["actual_start"]
        end_time = session["actual_end"] if session["actual_end"] else base_time
        
        # Each room has multiple cameras
        for i, position in enumerate(camera_positions[:random.randint(2, 4)]):
            camera_id = f"{session['room_id']}-CAM-{i+1}"
            
            duration = int((end_time - start_time).total_seconds()) if session["actual_end"] else None
            file_size = (duration * 2.5 / 60) if duration else None  # ~2.5 MB per minute
            
            # AI-detected events (simulated)
            detected_events = []
            if random.random() < 0.3:
                detected_events.append("Hand hygiene compliance verified")
            if random.random() < 0.2:
                detected_events.append("Surgical count completed")
            if random.random() < 0.1:
                detected_events.append("Equipment handoff detected")
            if random.random() < 0.05:
                detected_events.append("Potential safety concern flagged")
            
            video = {
                "video_id": f"VID-{random.randint(100000, 999999)}",
                "room_id": session["room_id"],
                "session_id": session["session_id"],
                "camera_id": camera_id,
                "camera_position": position,
                "recording_start": start_time,
                "recording_end": session["actual_end"],
                "duration_seconds": duration,
                "resolution": "1920x1080",
                "fps": 30,
                "file_path": f"/Volumes/{CATALOG}/{DIGITAL_TWIN_SCHEMA}/video_recordings/{session['session_id']}/{camera_id}.mp4" if duration else None,
                "file_size_mb": round(file_size, 2) if file_size else None,
                "status": "completed" if session["actual_end"] else "recording",
                "ai_analysis_complete": random.choice([True, False]) if session["actual_end"] else False,
                "detected_events": detected_events
            }
            video_records.append(video)

# Create video metadata DataFrame
video_df = spark.createDataFrame(video_records, video_metadata_schema)
video_df.write.mode("overwrite").saveAsTable(VIDEO_METADATA_TABLE)

print(f"Generated {len(video_records)} video records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Unified Digital Twin View
# MAGIC 
# MAGIC Join all data sources into a comprehensive digital twin view

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create comprehensive digital twin view
# MAGIC CREATE OR REPLACE VIEW lab_of_the_future.digital_twin.surgery_room_twin AS
# MAGIC WITH latest_state AS (
# MAGIC   SELECT *,
# MAGIC     ROW_NUMBER() OVER (PARTITION BY room_id ORDER BY timestamp DESC) as rn
# MAGIC   FROM lab_of_the_future.digital_twin.room_state
# MAGIC )
# MAGIC SELECT 
# MAGIC   r.room_id,
# MAGIC   r.room_name,
# MAGIC   r.room_type,
# MAGIC   r.floor,
# MAGIC   r.building,
# MAGIC   s.session_id,
# MAGIC   s.patient_id,
# MAGIC   s.procedure_type,
# MAGIC   s.lead_surgeon,
# MAGIC   s.status as session_status,
# MAGIC   s.actual_start,
# MAGIC   ls.timestamp as state_timestamp,
# MAGIC   ls.patient_heart_rate,
# MAGIC   ls.patient_heart_rate_variability,
# MAGIC   ls.patient_status,
# MAGIC   ls.active_imaging_modality,
# MAGIC   ls.images_taken,
# MAGIC   ls.video_feed_active,
# MAGIC   ls.personnel_count,
# MAGIC   ls.room_temperature_c,
# MAGIC   ls.room_humidity_pct,
# MAGIC   ls.alerts_active,
# MAGIC   (SELECT COUNT(*) FROM lab_of_the_future.digital_twin.video_metadata v 
# MAGIC    WHERE v.session_id = s.session_id AND v.status = 'recording') as active_cameras
# MAGIC FROM lab_of_the_future.digital_twin.surgery_rooms r
# MAGIC LEFT JOIN lab_of_the_future.digital_twin.surgery_sessions s 
# MAGIC   ON r.room_id = s.room_id AND s.status = 'in_progress'
# MAGIC LEFT JOIN latest_state ls 
# MAGIC   ON r.room_id = ls.room_id AND ls.rn = 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- View current state of all surgery rooms
# MAGIC SELECT * FROM lab_of_the_future.digital_twin.surgery_room_twin;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Link EKG Data to Digital Twin
# MAGIC 
# MAGIC Connect real-time EKG data from Notebook 1 to the digital twin

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create view joining EKG data with surgery sessions
# MAGIC CREATE OR REPLACE VIEW lab_of_the_future.digital_twin.patient_vitals_timeline AS
# MAGIC SELECT 
# MAGIC   s.session_id,
# MAGIC   s.room_id,
# MAGIC   s.patient_id,
# MAGIC   s.procedure_type,
# MAGIC   s.lead_surgeon,
# MAGIC   e.recording_datetime,
# MAGIC   e.heart_rate_bpm,
# MAGIC   e.device_id,
# MAGIC   e.lead_count,
# MAGIC   CASE 
# MAGIC     WHEN e.heart_rate_bpm > 130 OR e.heart_rate_bpm < 45 THEN 'CRITICAL'
# MAGIC     WHEN e.heart_rate_bpm > 110 OR e.heart_rate_bpm < 55 THEN 'WARNING'
# MAGIC     ELSE 'NORMAL'
# MAGIC   END as vital_status
# MAGIC FROM lab_of_the_future.digital_twin.surgery_sessions s
# MAGIC INNER JOIN lab_of_the_future.healthcare_data.ekg_silver e
# MAGIC   ON s.patient_id = e.patient_id
# MAGIC   AND e.recording_datetime BETWEEN s.actual_start AND COALESCE(s.actual_end, current_timestamp())
# MAGIC ORDER BY s.session_id, e.recording_datetime;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Link DICOM Images to Digital Twin
# MAGIC 
# MAGIC Connect imaging data from Notebook 2 to the digital twin

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create view linking DICOM images to surgery sessions
# MAGIC CREATE OR REPLACE VIEW lab_of_the_future.digital_twin.surgery_imaging AS
# MAGIC SELECT 
# MAGIC   s.session_id,
# MAGIC   s.room_id,
# MAGIC   s.patient_id,
# MAGIC   s.procedure_type,
# MAGIC   m.modality,
# MAGIC   m.study_instance_uid,
# MAGIC   m.series_description,
# MAGIC   m.body_part_examined,
# MAGIC   m.acquisition_datetime,
# MAGIC   f.mean_intensity,
# MAGIC   f.histogram_entropy,
# MAGIC   f.edge_density,
# MAGIC   f.file_path
# MAGIC FROM lab_of_the_future.digital_twin.surgery_sessions s
# MAGIC INNER JOIN lab_of_the_future.healthcare_data.dicom_metadata m
# MAGIC   ON s.patient_id = m.patient_id
# MAGIC LEFT JOIN lab_of_the_future.healthcare_data.dicom_features f
# MAGIC   ON m.sop_instance_uid = f.file_name
# MAGIC ORDER BY s.session_id, m.acquisition_datetime;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Alerting System
# MAGIC 
# MAGIC Create an alerting table for critical events detected by the digital twin

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create alerts table
# MAGIC CREATE TABLE IF NOT EXISTS lab_of_the_future.digital_twin.alerts (
# MAGIC   alert_id STRING,
# MAGIC   room_id STRING,
# MAGIC   session_id STRING,
# MAGIC   alert_type STRING,
# MAGIC   severity STRING,
# MAGIC   message STRING,
# MAGIC   source_system STRING,
# MAGIC   triggered_at TIMESTAMP,
# MAGIC   acknowledged_at TIMESTAMP,
# MAGIC   acknowledged_by STRING,
# MAGIC   resolved_at TIMESTAMP,
# MAGIC   resolution_notes STRING
# MAGIC )
# MAGIC COMMENT 'Digital twin alerts for surgery room monitoring';

# COMMAND ----------

# Generate sample alerts from room state data
alerts_data = []
states_df_local = spark.table(ROOM_STATE_TABLE).collect()

for state in states_df_local:
    if state.alerts_active:
        for alert_msg in state.alerts_active:
            severity = "CRITICAL" if "CRITICAL" in alert_msg else ("WARNING" if "WARNING" in alert_msg else "INFO")
            alert = {
                "alert_id": f"ALERT-{random.randint(100000, 999999)}",
                "room_id": state.room_id,
                "session_id": state.session_id,
                "alert_type": "vital_sign" if "heart rate" in alert_msg.lower() else "equipment",
                "severity": severity,
                "message": alert_msg,
                "source_system": "digital_twin_engine",
                "triggered_at": state.timestamp,
                "acknowledged_at": state.timestamp + timedelta(minutes=random.randint(1, 5)) if random.random() > 0.3 else None,
                "acknowledged_by": random.choice(nurses) if random.random() > 0.3 else None,
                "resolved_at": state.timestamp + timedelta(minutes=random.randint(5, 30)) if random.random() > 0.5 else None,
                "resolution_notes": "Condition stabilized" if random.random() > 0.5 else None
            }
            alerts_data.append(alert)

if alerts_data:
    alerts_df = spark.createDataFrame(alerts_data)
    alerts_df.write.mode("append").saveAsTable(ALERTS_TABLE)
    print(f"Generated {len(alerts_data)} alerts")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Digital Twin Analytics Dashboard Data

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Summary statistics for digital twin dashboard
# MAGIC SELECT 
# MAGIC   'Active Surgeries' as metric,
# MAGIC   CAST(COUNT(DISTINCT session_id) AS STRING) as value
# MAGIC FROM lab_of_the_future.digital_twin.surgery_sessions
# MAGIC WHERE status = 'in_progress'
# MAGIC 
# MAGIC UNION ALL
# MAGIC 
# MAGIC SELECT 
# MAGIC   'Rooms Occupied',
# MAGIC   CAST(COUNT(DISTINCT room_id) AS STRING)
# MAGIC FROM lab_of_the_future.digital_twin.surgery_sessions
# MAGIC WHERE status = 'in_progress'
# MAGIC 
# MAGIC UNION ALL
# MAGIC 
# MAGIC SELECT 
# MAGIC   'Active Alerts',
# MAGIC   CAST(COUNT(*) AS STRING)
# MAGIC FROM lab_of_the_future.digital_twin.alerts
# MAGIC WHERE resolved_at IS NULL
# MAGIC 
# MAGIC UNION ALL
# MAGIC 
# MAGIC SELECT 
# MAGIC   'Critical Alerts',
# MAGIC   CAST(COUNT(*) AS STRING)
# MAGIC FROM lab_of_the_future.digital_twin.alerts
# MAGIC WHERE resolved_at IS NULL AND severity = 'CRITICAL'
# MAGIC 
# MAGIC UNION ALL
# MAGIC 
# MAGIC SELECT 
# MAGIC   'Surgeries Today',
# MAGIC   CAST(COUNT(*) AS STRING)
# MAGIC FROM lab_of_the_future.digital_twin.surgery_sessions
# MAGIC WHERE DATE(scheduled_start) = CURRENT_DATE()
# MAGIC 
# MAGIC UNION ALL
# MAGIC 
# MAGIC SELECT
# MAGIC   'Active Video Feeds',
# MAGIC   CAST(COUNT(*) AS STRING)
# MAGIC FROM lab_of_the_future.digital_twin.video_metadata
# MAGIC WHERE status = 'recording';

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Room utilization by day
# MAGIC SELECT 
# MAGIC   DATE(scheduled_start) as surgery_date,
# MAGIC   room_id,
# MAGIC   COUNT(*) as surgeries,
# MAGIC   SUM(TIMESTAMPDIFF(HOUR, actual_start, COALESCE(actual_end, current_timestamp()))) as total_hours
# MAGIC FROM lab_of_the_future.digital_twin.surgery_sessions
# MAGIC WHERE actual_start IS NOT NULL
# MAGIC GROUP BY DATE(scheduled_start), room_id
# MAGIC ORDER BY surgery_date DESC, room_id;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Real-Time Streaming Setup (Production)
# MAGIC 
# MAGIC In production, you would set up streaming pipelines to continuously update the digital twin state.
# MAGIC Below is an example of how this would be configured:

# COMMAND ----------

# Example: Streaming query for real-time room state updates (not executed in demo)
"""
# Real-time EKG streaming to digital twin
ekg_stream = (
    spark.readStream
    .format("delta")
    .table("lab_of_the_future.healthcare_data.ekg_silver")
)

# Join with active sessions and update room state
room_state_updates = (
    ekg_stream
    .join(
        spark.table("lab_of_the_future.digital_twin.surgery_sessions")
        .filter("status = 'in_progress'"),
        "patient_id"
    )
    .select(
        "room_id",
        "session_id",
        "recording_datetime",
        "heart_rate_bpm",
        expr("CASE WHEN heart_rate_bpm > 130 THEN 'critical' WHEN heart_rate_bpm > 110 THEN 'warning' ELSE 'stable' END as patient_status")
    )
)

# Write updates to room state
room_state_updates.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", "/Volumes/lab_of_the_future/digital_twin/_checkpoints/room_state")
    .toTable("lab_of_the_future.digital_twin.room_state_stream")
"""

print("Streaming setup code provided as reference for production deployment")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC Continue to **Notebook 4** to learn how to:
# MAGIC - Build a Databricks App for the Lab of the Future
# MAGIC - Create interactive dashboards combining EKG and DICOM data
# MAGIC - Deploy the application for clinical users
# MAGIC 
# MAGIC The digital twin infrastructure created in this notebook provides the foundation for 
# MAGIC real-time monitoring and analytics applications.
