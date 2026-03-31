-- Patient summary with vital statistics
SELECT 
  patient_id,
  ekg_recordings,
  avg_heart_rate,
  min_heart_rate,
  max_heart_rate,
  imaging_studies,
  surgeries,
  first_recording,
  last_recording
FROM lab_of_the_future.healthcare_data.app_patient_summary
ORDER BY last_recording DESC
LIMIT 100
