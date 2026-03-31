-- Current status of all surgery rooms
SELECT 
  room_id,
  room_name,
  room_type,
  floor,
  current_status,
  patient_id,
  procedure_type,
  lead_surgeon,
  patient_heart_rate,
  patient_status,
  personnel_count,
  video_feed_active,
  room_temperature_c,
  last_update,
  active_alert_count
FROM lab_of_the_future.digital_twin.app_room_status
ORDER BY 
  CASE current_status 
    WHEN 'in_progress' THEN 1 
    WHEN 'scheduled' THEN 2 
    ELSE 3 
  END,
  room_id
