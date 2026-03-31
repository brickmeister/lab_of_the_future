-- Imaging history for a specific patient
-- Parameters: :patient_id
SELECT 
  study_instance_uid,
  modality,
  study_description,
  study_date,
  manufacturer,
  file_path
FROM lab_of_the_future.healthcare_data.dicom_metadata
WHERE patient_id = :patient_id
ORDER BY study_date DESC
