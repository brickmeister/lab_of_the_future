-- Active and recent alerts
SELECT 
  alert_id,
  room_name,
  patient_id,
  procedure_type,
  severity,
  message,
  alert_status,
  triggered_at,
  duration_minutes
FROM lab_of_the_future.digital_twin.app_alerts_dashboard
WHERE alert_status != 'resolved'
   OR triggered_at > DATEADD(HOUR, -24, current_timestamp())
ORDER BY 
  CASE severity WHEN 'CRITICAL' THEN 1 WHEN 'WARNING' THEN 2 ELSE 3 END,
  triggered_at DESC
LIMIT 50
