-- Daily operational metrics for dashboard KPIs
SELECT 
  DATE(scheduled_start) as date,
  COUNT(*) as total_surgeries,
  COUNT(DISTINCT patient_id) as unique_patients,
  COUNT(DISTINCT room_id) as rooms_used,
  AVG(TIMESTAMPDIFF(MINUTE, actual_start, actual_end)) as avg_duration_minutes,
  SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
  SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) as cancelled
FROM lab_of_the_future.digital_twin.surgery_sessions
WHERE scheduled_start >= DATEADD(DAY, -30, current_date())
GROUP BY DATE(scheduled_start)
ORDER BY date DESC
