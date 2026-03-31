# Databricks notebook source
# MAGIC %md
# MAGIC # Lab of the Future Workshop - Notebook 5
# MAGIC ## Delta Sharing for Healthcare Data Collaboration
# MAGIC 
# MAGIC This notebook demonstrates how to set up **Delta Sharing** to securely share:
# MAGIC - DICOM image data volumes
# MAGIC - Featurized imaging results
# MAGIC - Digital twin analytics data
# MAGIC 
# MAGIC ### What is Delta Sharing?
# MAGIC Delta Sharing is an open protocol for secure real-time exchange of large datasets.
# MAGIC It enables sharing data across organizations without copying or moving the data.
# MAGIC 
# MAGIC ### Learning Objectives
# MAGIC - Configure Delta Sharing providers and recipients
# MAGIC - Create shares for healthcare data
# MAGIC - Set up secure access for external collaborators
# MAGIC - Monitor sharing activity and access patterns
# MAGIC 
# MAGIC ### Use Cases for Healthcare Data Sharing
# MAGIC - **Research Collaborations**: Share de-identified imaging data with research institutions
# MAGIC - **Partner Integration**: Enable partner hospitals to access shared analytics
# MAGIC - **Vendor Data Exchange**: Share operational data with equipment vendors
# MAGIC - **Regulatory Reporting**: Provide auditors access to compliance data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Architecture Overview
# MAGIC 
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                         Delta Sharing Architecture                           │
# MAGIC ├─────────────────────────────────────────────────────────────────────────────┤
# MAGIC │                                                                             │
# MAGIC │  ┌─────────────────────────────┐         ┌─────────────────────────────┐   │
# MAGIC │  │     Provider (Your Org)     │         │    Recipients (Partners)    │   │
# MAGIC │  ├─────────────────────────────┤         ├─────────────────────────────┤   │
# MAGIC │  │                             │         │                             │   │
# MAGIC │  │  ┌─────────────────────┐    │         │  ┌─────────────────────┐    │   │
# MAGIC │  │  │   Unity Catalog     │    │  Share  │  │  Research Hospital  │    │   │
# MAGIC │  │  │   - DICOM Images    │────┼────────►│  │  - Spark/Pandas     │    │   │
# MAGIC │  │  │   - Features        │    │         │  │  - Delta Sharing    │    │   │
# MAGIC │  │  │   - Analytics       │    │         │  │    Connector        │    │   │
# MAGIC │  │  └─────────────────────┘    │         │  └─────────────────────┘    │   │
# MAGIC │  │                             │         │                             │   │
# MAGIC │  │  ┌─────────────────────┐    │         │  ┌─────────────────────┐    │   │
# MAGIC │  │  │   Delta Sharing     │    │  Share  │  │  Equipment Vendor   │    │   │
# MAGIC │  │  │   Server            │────┼────────►│  │  - Python Client    │    │   │
# MAGIC │  │  │   (Built into UC)   │    │         │  │  - REST API         │    │   │
# MAGIC │  │  └─────────────────────┘    │         │  └─────────────────────┘    │   │
# MAGIC │  │                             │         │                             │   │
# MAGIC │  └─────────────────────────────┘         └─────────────────────────────┘   │
# MAGIC │                                                                             │
# MAGIC └─────────────────────────────────────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

CATALOG = "lab_of_the_future"
SCHEMA_HEALTHCARE = "healthcare_data"
SCHEMA_DIGITAL_TWIN = "digital_twin"

# Share names
IMAGING_SHARE = "lab_future_imaging_share"
ANALYTICS_SHARE = "lab_future_analytics_share"
RESEARCH_SHARE = "lab_future_research_share"

print(f"Catalog: {CATALOG}")
print(f"Shares to create: {IMAGING_SHARE}, {ANALYTICS_SHARE}, {RESEARCH_SHARE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Create Delta Shares
# MAGIC 
# MAGIC Create shares to organize the data you want to share with external parties

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create share for medical imaging data
# MAGIC CREATE SHARE IF NOT EXISTS lab_future_imaging_share
# MAGIC COMMENT 'Medical imaging data share for research collaborations - includes DICOM metadata and image features';

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create share for analytics and digital twin data
# MAGIC CREATE SHARE IF NOT EXISTS lab_future_analytics_share
# MAGIC COMMENT 'Operational analytics share for partner hospitals - includes aggregated metrics and trends';

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create share for de-identified research data
# MAGIC CREATE SHARE IF NOT EXISTS lab_future_research_share
# MAGIC COMMENT 'De-identified research dataset for academic collaborations';

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Add Tables to Shares
# MAGIC 
# MAGIC Add specific tables and views to each share

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imaging Share - DICOM Data

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Add DICOM metadata to imaging share
# MAGIC ALTER SHARE lab_future_imaging_share
# MAGIC ADD TABLE lab_of_the_future.healthcare_data.dicom_metadata
# MAGIC COMMENT 'DICOM file metadata including modality, body part, and acquisition details';

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Add image embeddings to imaging share
# MAGIC ALTER SHARE lab_future_imaging_share
# MAGIC ADD TABLE lab_of_the_future.healthcare_data.dicom_embeddings
# MAGIC COMMENT 'Image embedding vectors for ML model training';

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analytics Share - Operational Data

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Add aggregated EKG metrics (no raw patient data)
# MAGIC ALTER SHARE lab_future_analytics_share
# MAGIC ADD TABLE lab_of_the_future.healthcare_data.ekg_gold_daily_summary
# MAGIC COMMENT 'Daily aggregated EKG monitoring statistics';

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create and add de-identified surgery metrics view
# MAGIC CREATE OR REPLACE VIEW lab_of_the_future.digital_twin.surgery_metrics_deidentified AS
# MAGIC SELECT 
# MAGIC   DATE(scheduled_start) as surgery_date,
# MAGIC   room_type,
# MAGIC   procedure_type,
# MAGIC   COUNT(*) as procedure_count,
# MAGIC   AVG(TIMESTAMPDIFF(MINUTE, actual_start, actual_end)) as avg_duration_minutes,
# MAGIC   SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_count,
# MAGIC   SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) as cancelled_count
# MAGIC FROM lab_of_the_future.digital_twin.surgery_sessions s
# MAGIC JOIN lab_of_the_future.digital_twin.surgery_rooms r ON s.room_id = r.room_id
# MAGIC GROUP BY DATE(scheduled_start), room_type, procedure_type;

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER SHARE lab_future_analytics_share
# MAGIC ADD TABLE lab_of_the_future.digital_twin.surgery_metrics_deidentified
# MAGIC COMMENT 'De-identified surgery metrics by procedure type and date';

# COMMAND ----------

# MAGIC %md
# MAGIC ### Research Share - De-identified Data for Academic Use

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create de-identified research dataset
# MAGIC CREATE OR REPLACE VIEW lab_of_the_future.healthcare_data.research_imaging_dataset AS
# MAGIC SELECT 
# MAGIC   -- Use hashed patient ID for de-identification
# MAGIC   SHA2(patient_id, 256) as patient_hash,
# MAGIC   modality,
# MAGIC   study_description,
# MAGIC   patient_sex,
# MAGIC   manufacturer,
# MAGIC   study_date
# MAGIC FROM lab_of_the_future.healthcare_data.dicom_metadata;

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER SHARE lab_future_research_share
# MAGIC ADD TABLE lab_of_the_future.healthcare_data.research_imaging_dataset
# MAGIC COMMENT 'De-identified imaging dataset for academic research - HIPAA compliant';

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Add de-identified EKG research data
# MAGIC CREATE OR REPLACE VIEW lab_of_the_future.healthcare_data.research_ekg_dataset AS
# MAGIC SELECT 
# MAGIC   SHA2(patient_id, 256) as patient_hash,
# MAGIC   DATE(recording_datetime) as recording_date,
# MAGIC   heart_rate_bpm,
# MAGIC   lead_count,
# MAGIC   device_manufacturer,
# MAGIC   device_model
# MAGIC FROM lab_of_the_future.healthcare_data.ekg_silver;

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER SHARE lab_future_research_share
# MAGIC ADD TABLE lab_of_the_future.healthcare_data.research_ekg_dataset
# MAGIC COMMENT 'De-identified EKG dataset for cardiac research';

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: View Share Contents
# MAGIC 
# MAGIC Verify what data is included in each share

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Show all shares
# MAGIC SHOW SHARES;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Show contents of imaging share
# MAGIC DESCRIBE SHARE lab_future_imaging_share;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Show contents of analytics share
# MAGIC DESCRIBE SHARE lab_future_analytics_share;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Show contents of research share
# MAGIC DESCRIBE SHARE lab_future_research_share;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Create Recipients
# MAGIC 
# MAGIC Recipients are external organizations that will receive access to shared data

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create recipient for research hospital partner
# MAGIC CREATE RECIPIENT IF NOT EXISTS research_hospital_partner
# MAGIC COMMENT 'University Medical Research Center - Academic research collaboration';

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create recipient for equipment vendor
# MAGIC CREATE RECIPIENT IF NOT EXISTS imaging_equipment_vendor
# MAGIC COMMENT 'MedTech Imaging Inc - Equipment analytics and optimization';

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create recipient for regulatory auditor
# MAGIC CREATE RECIPIENT IF NOT EXISTS regulatory_auditor
# MAGIC COMMENT 'Healthcare Compliance Audit - Annual regulatory review';

# COMMAND ----------

# MAGIC %sql
# MAGIC -- List all recipients
# MAGIC SHOW RECIPIENTS;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Grant Share Access to Recipients
# MAGIC 
# MAGIC Assign specific shares to recipients based on their data needs

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Grant research share to research hospital (academic use)
# MAGIC GRANT SELECT ON SHARE lab_future_research_share TO RECIPIENT research_hospital_partner;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Grant imaging share to equipment vendor (for ML model improvement)
# MAGIC GRANT SELECT ON SHARE lab_future_imaging_share TO RECIPIENT imaging_equipment_vendor;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Grant analytics share to equipment vendor (for operational insights)
# MAGIC GRANT SELECT ON SHARE lab_future_analytics_share TO RECIPIENT imaging_equipment_vendor;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Grant analytics share to regulatory auditor
# MAGIC GRANT SELECT ON SHARE lab_future_analytics_share TO RECIPIENT regulatory_auditor;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Generate Activation Links
# MAGIC 
# MAGIC Recipients need activation links to access the shared data

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Show recipient details including activation link
# MAGIC DESCRIBE RECIPIENT research_hospital_partner;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Activation Link Distribution
# MAGIC 
# MAGIC The activation link and credentials should be securely shared with recipients:
# MAGIC 
# MAGIC 1. **Download the credentials file** from the recipient details
# MAGIC 2. **Securely transmit** the file to the recipient organization
# MAGIC 3. Recipients use the credentials to connect via:
# MAGIC    - Databricks Delta Sharing connector
# MAGIC    - Python `delta-sharing` library
# MAGIC    - Apache Spark with Delta Sharing support
# MAGIC    - Pandas via `delta-sharing` package

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Recipient Access Instructions
# MAGIC 
# MAGIC Provide these instructions to your data recipients

# COMMAND ----------

# MAGIC %md
# MAGIC ### For Python/Pandas Recipients
# MAGIC 
# MAGIC ```python
# MAGIC # Install the delta-sharing package
# MAGIC # pip install delta-sharing
# MAGIC 
# MAGIC import delta_sharing
# MAGIC 
# MAGIC # Path to the profile file (credentials downloaded from provider)
# MAGIC profile_file = "/path/to/config.share"
# MAGIC 
# MAGIC # Create a sharing client
# MAGIC client = delta_sharing.SharingClient(profile_file)
# MAGIC 
# MAGIC # List available shares
# MAGIC shares = client.list_shares()
# MAGIC for share in shares:
# MAGIC     print(f"Share: {share.name}")
# MAGIC 
# MAGIC # List tables in a share
# MAGIC tables = client.list_all_tables()
# MAGIC for table in tables:
# MAGIC     print(f"Table: {table.share}.{table.schema}.{table.name}")
# MAGIC 
# MAGIC # Load a shared table into Pandas
# MAGIC table_url = f"{profile_file}#lab_future_research_share.default.research_imaging_dataset"
# MAGIC df = delta_sharing.load_as_pandas(table_url)
# MAGIC print(df.head())
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### For Spark Recipients
# MAGIC 
# MAGIC ```python
# MAGIC # In Databricks or Spark environment
# MAGIC from pyspark.sql import SparkSession
# MAGIC 
# MAGIC spark = SparkSession.builder \
# MAGIC     .appName("DeltaSharingReader") \
# MAGIC     .config("spark.jars.packages", "io.delta:delta-sharing-spark_2.12:0.6.0") \
# MAGIC     .getOrCreate()
# MAGIC 
# MAGIC # Load shared table
# MAGIC df = spark.read.format("deltaSharing") \
# MAGIC     .option("profileFile", "/path/to/config.share") \
# MAGIC     .load("lab_future_research_share.default.research_imaging_dataset")
# MAGIC 
# MAGIC df.show()
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### For Databricks Recipients
# MAGIC 
# MAGIC ```sql
# MAGIC -- Create a catalog from the share
# MAGIC CREATE CATALOG IF NOT EXISTS research_data_incoming
# MAGIC USING SHARE `provider_metastore`.lab_future_research_share;
# MAGIC 
# MAGIC -- Query shared tables
# MAGIC SELECT * FROM research_data_incoming.default.research_imaging_dataset LIMIT 10;
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Sharing Volumes (DICOM Image Files)
# MAGIC 
# MAGIC Delta Sharing also supports sharing Unity Catalog Volumes containing files

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Note: Volume sharing requires Unity Catalog with volume sharing enabled
# MAGIC -- Create a share for the DICOM image volume
# MAGIC CREATE SHARE IF NOT EXISTS lab_future_dicom_volume_share
# MAGIC COMMENT 'DICOM image files volume share for research collaborations';

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Add the volume to the share (if volume sharing is enabled)
# MAGIC -- ALTER SHARE lab_future_dicom_volume_share
# MAGIC -- ADD VOLUME lab_of_the_future.healthcare_data.dicom_raw;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Alternative: Share Volume Metadata
# MAGIC 
# MAGIC If direct volume sharing is not available, share metadata with file paths:

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create a view with volume file references
# MAGIC CREATE OR REPLACE VIEW lab_of_the_future.healthcare_data.dicom_file_catalog AS
# MAGIC SELECT 
# MAGIC   file_path,
# MAGIC   patient_id,
# MAGIC   study_instance_uid,
# MAGIC   modality,
# MAGIC   study_description,
# MAGIC   study_date
# MAGIC FROM lab_of_the_future.healthcare_data.dicom_metadata;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Add file catalog to imaging share
# MAGIC ALTER SHARE lab_future_imaging_share
# MAGIC ADD TABLE lab_of_the_future.healthcare_data.dicom_file_catalog
# MAGIC COMMENT 'Catalog of DICOM files with paths and metadata';

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Monitor Sharing Activity
# MAGIC 
# MAGIC Track who is accessing shared data and when

# COMMAND ----------

# MAGIC %sql
# MAGIC -- View sharing audit logs (requires appropriate permissions)
# MAGIC -- This query would work with system tables if enabled
# MAGIC 
# MAGIC -- SELECT 
# MAGIC --   event_time,
# MAGIC --   recipient_name,
# MAGIC --   share_name,
# MAGIC --   action_name,
# MAGIC --   request_id
# MAGIC -- FROM system.access.audit
# MAGIC -- WHERE service_name = 'deltasharing'
# MAGIC -- ORDER BY event_time DESC
# MAGIC -- LIMIT 100;
# MAGIC 
# MAGIC SELECT 'Audit logging available via system tables when enabled' as note;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: Manage Share Permissions

# COMMAND ----------

# MAGIC %sql
# MAGIC -- View current grants on a share
# MAGIC SHOW GRANTS ON SHARE lab_future_research_share;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Revoke access if needed (example)
# MAGIC -- REVOKE SELECT ON SHARE lab_future_research_share FROM RECIPIENT research_hospital_partner;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Governance Best Practices
# MAGIC 
# MAGIC ### 1. De-identification
# MAGIC - Always hash or remove direct patient identifiers
# MAGIC - Use age ranges instead of exact dates
# MAGIC - Remove location-specific information
# MAGIC 
# MAGIC ### 2. Access Controls
# MAGIC - Implement least-privilege access
# MAGIC - Regular access reviews
# MAGIC - Time-limited access for auditors
# MAGIC 
# MAGIC ### 3. Audit and Compliance
# MAGIC - Enable audit logging
# MAGIC - Document data sharing agreements
# MAGIC - Regular compliance reviews

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Example: Create a time-limited access view
# MAGIC CREATE OR REPLACE VIEW lab_of_the_future.healthcare_data.audit_limited_view AS
# MAGIC SELECT *
# MAGIC FROM lab_of_the_future.healthcare_data.research_imaging_dataset
# MAGIC WHERE acquisition_date >= DATE_SUB(current_date(), 365);  -- Last year only

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC 
# MAGIC In this notebook, we configured Delta Sharing for the Lab of the Future platform:
# MAGIC 
# MAGIC | Share | Data Included | Recipients |
# MAGIC |-------|---------------|------------|
# MAGIC | `lab_future_imaging_share` | DICOM metadata, features, embeddings, file catalog | Equipment vendors |
# MAGIC | `lab_future_analytics_share` | Operational metrics, surgery statistics | Vendors, auditors |
# MAGIC | `lab_future_research_share` | De-identified imaging & EKG data | Research hospitals |
# MAGIC 
# MAGIC ### Key Takeaways
# MAGIC 
# MAGIC 1. **Delta Sharing** enables secure data sharing without copying data
# MAGIC 2. **Unity Catalog** provides centralized governance for shared data
# MAGIC 3. **De-identification** is critical for healthcare data sharing compliance
# MAGIC 4. **Recipients** can access data using standard tools (Python, Spark, etc.)
# MAGIC 5. **Audit logging** helps maintain compliance and security

# COMMAND ----------

# MAGIC %md
# MAGIC ## Workshop Complete!
# MAGIC 
# MAGIC Congratulations on completing the **Lab of the Future Workshop**!
# MAGIC 
# MAGIC ### What You Learned
# MAGIC 
# MAGIC 1. **Notebook 1**: Ingesting EKG data with Spark Declarative Pipelines (DLT)
# MAGIC 2. **Notebook 2**: Processing DICOM images with the Pixels accelerator
# MAGIC 3. **Notebook 3**: Building a surgery room digital twin
# MAGIC 4. **Notebook 4**: Creating healthcare dashboards with Databricks Apps
# MAGIC 5. **Notebook 5**: Setting up Delta Sharing for secure data collaboration
# MAGIC 
# MAGIC ### Next Steps
# MAGIC 
# MAGIC - Deploy the DLT pipeline to production
# MAGIC - Connect real EKG devices and PACS systems
# MAGIC - Implement real-time streaming for the digital twin
# MAGIC - Onboard actual research partners via Delta Sharing
# MAGIC - Add ML models for predictive analytics
# MAGIC 
# MAGIC ### Resources
# MAGIC 
# MAGIC - [Databricks Delta Live Tables Documentation](https://docs.databricks.com/delta-live-tables/)
# MAGIC - [Databricks Pixels Accelerator](https://docs.databricks.com/machine-learning/pixels.html)
# MAGIC - [Delta Sharing Documentation](https://docs.databricks.com/data-sharing/)
# MAGIC - [Databricks Apps Documentation](https://docs.databricks.com/dev-tools/databricks-apps/)
# MAGIC - [Unity Catalog Documentation](https://docs.databricks.com/data-governance/unity-catalog/)
