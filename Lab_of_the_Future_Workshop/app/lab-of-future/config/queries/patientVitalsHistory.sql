-- Heart rate history for a specific patient
-- Parameters: :patient_id
SELECT 
  recording_datetime,
  heart_rate_bpm,
  device_id
FROM lab_of_the_future.healthcare_data.ekg_silver
WHERE patient_id = :patient_id
ORDER BY recording_datetime DESC
LIMIT 500
