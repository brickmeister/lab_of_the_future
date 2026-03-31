# Databricks notebook source
# MAGIC %md
# MAGIC # Lab of the Future Workshop - Notebook 4
# MAGIC ## Building a Healthcare Dashboard with Databricks Apps
# MAGIC 
# MAGIC This notebook demonstrates how to build a **Databricks App** that combines:
# MAGIC - EKG vital sign data from Notebook 1
# MAGIC - DICOM medical imaging data from Notebook 2
# MAGIC - Digital twin data from Notebook 3
# MAGIC 
# MAGIC ### Learning Objectives
# MAGIC - Understand Databricks Apps architecture
# MAGIC - Create SQL queries for healthcare analytics
# MAGIC - Build interactive dashboards with AppKit
# MAGIC - Deploy the application for clinical users
# MAGIC 
# MAGIC ### Application Features
# MAGIC - **Patient Overview**: Real-time vitals and imaging history
# MAGIC - **Surgery Room Monitoring**: Digital twin status dashboard
# MAGIC - **Alert Management**: Critical event tracking and response
# MAGIC - **Analytics**: Trends and operational insights

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites
# MAGIC 
# MAGIC Before building the app, ensure you have:
# MAGIC 1. Completed Notebooks 1-3 to populate the data
# MAGIC 2. Databricks CLI installed and authenticated
# MAGIC 3. Access to a SQL Warehouse

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Verify Data Availability
# MAGIC 
# MAGIC Check that all required tables exist and contain data

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Verify data sources
# MAGIC SELECT 'EKG Silver' as source, COUNT(*) as records 
# MAGIC FROM lab_of_the_future.healthcare_data.ekg_silver
# MAGIC UNION ALL
# MAGIC SELECT 'DICOM Metadata', COUNT(*) 
# MAGIC FROM lab_of_the_future.healthcare_data.dicom_metadata
# MAGIC UNION ALL
# MAGIC SELECT 'Surgery Sessions', COUNT(*) 
# MAGIC FROM lab_of_the_future.digital_twin.surgery_sessions
# MAGIC UNION ALL
# MAGIC SELECT 'Room State', COUNT(*) 
# MAGIC FROM lab_of_the_future.digital_twin.room_state
# MAGIC UNION ALL
# MAGIC SELECT 'Alerts', COUNT(*) 
# MAGIC FROM lab_of_the_future.digital_twin.alerts;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Create App-Optimized Views
# MAGIC 
# MAGIC Create views that are optimized for the application's query patterns

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Patient summary view for the app
# MAGIC CREATE OR REPLACE VIEW lab_of_the_future.healthcare_data.app_patient_summary AS
# MAGIC SELECT 
# MAGIC   e.patient_id,
# MAGIC   COUNT(DISTINCT e.recording_datetime) as ekg_recordings,
# MAGIC   ROUND(AVG(e.heart_rate_bpm), 1) as avg_heart_rate,
# MAGIC   MIN(e.heart_rate_bpm) as min_heart_rate,
# MAGIC   MAX(e.heart_rate_bpm) as max_heart_rate,
# MAGIC   MIN(e.recording_datetime) as first_recording,
# MAGIC   MAX(e.recording_datetime) as last_recording,
# MAGIC   COALESCE(img.image_count, 0) as imaging_studies,
# MAGIC   COALESCE(surg.surgery_count, 0) as surgeries
# MAGIC FROM lab_of_the_future.healthcare_data.ekg_silver e
# MAGIC LEFT JOIN (
# MAGIC   SELECT patient_id, COUNT(*) as image_count
# MAGIC   FROM lab_of_the_future.healthcare_data.dicom_metadata
# MAGIC   GROUP BY patient_id
# MAGIC ) img ON e.patient_id = img.patient_id
# MAGIC LEFT JOIN (
# MAGIC   SELECT patient_id, COUNT(*) as surgery_count
# MAGIC   FROM lab_of_the_future.digital_twin.surgery_sessions
# MAGIC   GROUP BY patient_id
# MAGIC ) surg ON e.patient_id = surg.patient_id
# MAGIC GROUP BY e.patient_id, img.image_count, surg.surgery_count;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Room status view for real-time dashboard
# MAGIC CREATE OR REPLACE VIEW lab_of_the_future.digital_twin.app_room_status AS
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
# MAGIC   COALESCE(s.status, 'available') as current_status,
# MAGIC   s.patient_id,
# MAGIC   s.procedure_type,
# MAGIC   s.lead_surgeon,
# MAGIC   ls.patient_heart_rate,
# MAGIC   ls.patient_status,
# MAGIC   ls.personnel_count,
# MAGIC   ls.video_feed_active,
# MAGIC   ls.room_temperature_c,
# MAGIC   ls.timestamp as last_update,
# MAGIC   SIZE(ls.alerts_active) as active_alert_count
# MAGIC FROM lab_of_the_future.digital_twin.surgery_rooms r
# MAGIC LEFT JOIN lab_of_the_future.digital_twin.surgery_sessions s 
# MAGIC   ON r.room_id = s.room_id AND s.status = 'in_progress'
# MAGIC LEFT JOIN latest_state ls 
# MAGIC   ON r.room_id = ls.room_id AND ls.rn = 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Alert dashboard view
# MAGIC CREATE OR REPLACE VIEW lab_of_the_future.digital_twin.app_alerts_dashboard AS
# MAGIC SELECT 
# MAGIC   a.alert_id,
# MAGIC   a.room_id,
# MAGIC   r.room_name,
# MAGIC   a.session_id,
# MAGIC   s.patient_id,
# MAGIC   s.procedure_type,
# MAGIC   a.alert_type,
# MAGIC   a.severity,
# MAGIC   a.message,
# MAGIC   a.triggered_at,
# MAGIC   a.acknowledged_at,
# MAGIC   a.acknowledged_by,
# MAGIC   a.resolved_at,
# MAGIC   CASE 
# MAGIC     WHEN a.resolved_at IS NOT NULL THEN 'resolved'
# MAGIC     WHEN a.acknowledged_at IS NOT NULL THEN 'acknowledged'
# MAGIC     ELSE 'active'
# MAGIC   END as alert_status,
# MAGIC   TIMESTAMPDIFF(MINUTE, a.triggered_at, COALESCE(a.resolved_at, current_timestamp())) as duration_minutes
# MAGIC FROM lab_of_the_future.digital_twin.alerts a
# MAGIC LEFT JOIN lab_of_the_future.digital_twin.surgery_rooms r ON a.room_id = r.room_id
# MAGIC LEFT JOIN lab_of_the_future.digital_twin.surgery_sessions s ON a.session_id = s.session_id
# MAGIC ORDER BY a.triggered_at DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Initialize Databricks App
# MAGIC 
# MAGIC Use the Databricks CLI to scaffold a new AppKit application

# COMMAND ----------

# MAGIC %md
# MAGIC ### CLI Commands for App Creation
# MAGIC 
# MAGIC Run these commands in your terminal (not in the notebook):
# MAGIC 
# MAGIC ```bash
# MAGIC # 1. List available profiles
# MAGIC databricks auth profiles
# MAGIC 
# MAGIC # 2. Get default warehouse ID
# MAGIC databricks experimental aitools tools get-default-warehouse --profile <your-profile>
# MAGIC 
# MAGIC # 3. Create the app
# MAGIC databricks apps init \
# MAGIC   --name lab-of-future \
# MAGIC   --description "Lab of the Future Healthcare Dashboard" \
# MAGIC   --features analytics \
# MAGIC   --warehouse-id <warehouse-id> \
# MAGIC   --run none \
# MAGIC   --profile <your-profile>
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: SQL Query Files
# MAGIC 
# MAGIC The following SQL queries should be placed in `config/queries/` directory of your app.
# MAGIC Each query becomes available via the `queryKey` matching the filename.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query: `patientSummary.sql`
# MAGIC 
# MAGIC ```sql
# MAGIC -- Patient summary with vital statistics
# MAGIC SELECT 
# MAGIC   patient_id,
# MAGIC   ekg_recordings,
# MAGIC   avg_heart_rate,
# MAGIC   min_heart_rate,
# MAGIC   max_heart_rate,
# MAGIC   imaging_studies,
# MAGIC   surgeries,
# MAGIC   first_recording,
# MAGIC   last_recording
# MAGIC FROM lab_of_the_future.healthcare_data.app_patient_summary
# MAGIC ORDER BY last_recording DESC
# MAGIC LIMIT 100
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query: `roomStatus.sql`
# MAGIC 
# MAGIC ```sql
# MAGIC -- Current status of all surgery rooms
# MAGIC SELECT 
# MAGIC   room_id,
# MAGIC   room_name,
# MAGIC   room_type,
# MAGIC   floor,
# MAGIC   current_status,
# MAGIC   patient_id,
# MAGIC   procedure_type,
# MAGIC   lead_surgeon,
# MAGIC   patient_heart_rate,
# MAGIC   patient_status,
# MAGIC   personnel_count,
# MAGIC   video_feed_active,
# MAGIC   room_temperature_c,
# MAGIC   last_update,
# MAGIC   active_alert_count
# MAGIC FROM lab_of_the_future.digital_twin.app_room_status
# MAGIC ORDER BY 
# MAGIC   CASE current_status 
# MAGIC     WHEN 'in_progress' THEN 1 
# MAGIC     WHEN 'scheduled' THEN 2 
# MAGIC     ELSE 3 
# MAGIC   END,
# MAGIC   room_id
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query: `activeAlerts.sql`
# MAGIC 
# MAGIC ```sql
# MAGIC -- Active and recent alerts
# MAGIC SELECT 
# MAGIC   alert_id,
# MAGIC   room_name,
# MAGIC   patient_id,
# MAGIC   procedure_type,
# MAGIC   severity,
# MAGIC   message,
# MAGIC   alert_status,
# MAGIC   triggered_at,
# MAGIC   duration_minutes
# MAGIC FROM lab_of_the_future.digital_twin.app_alerts_dashboard
# MAGIC WHERE alert_status != 'resolved'
# MAGIC    OR triggered_at > DATEADD(HOUR, -24, current_timestamp())
# MAGIC ORDER BY 
# MAGIC   CASE severity WHEN 'CRITICAL' THEN 1 WHEN 'WARNING' THEN 2 ELSE 3 END,
# MAGIC   triggered_at DESC
# MAGIC LIMIT 50
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query: `patientVitalsHistory.sql`
# MAGIC 
# MAGIC ```sql
# MAGIC -- Heart rate history for a specific patient
# MAGIC -- Parameters: :patient_id
# MAGIC SELECT 
# MAGIC   recording_datetime,
# MAGIC   heart_rate_bpm,
# MAGIC   device_id
# MAGIC FROM lab_of_the_future.healthcare_data.ekg_silver
# MAGIC WHERE patient_id = :patient_id
# MAGIC ORDER BY recording_datetime DESC
# MAGIC LIMIT 500
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query: `imagingHistory.sql`
# MAGIC 
# MAGIC ```sql
# MAGIC -- Imaging history for a specific patient
# MAGIC -- Parameters: :patient_id
# MAGIC SELECT 
# MAGIC   study_instance_uid,
# MAGIC   modality,
# MAGIC   study_description,
# MAGIC   study_date,
# MAGIC   manufacturer,
# MAGIC   file_path
# MAGIC FROM lab_of_the_future.healthcare_data.dicom_metadata
# MAGIC WHERE patient_id = :patient_id
# MAGIC ORDER BY study_date DESC
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query: `operationalMetrics.sql`
# MAGIC 
# MAGIC ```sql
# MAGIC -- Daily operational metrics for dashboard KPIs
# MAGIC SELECT 
# MAGIC   DATE(scheduled_start) as date,
# MAGIC   COUNT(*) as total_surgeries,
# MAGIC   COUNT(DISTINCT patient_id) as unique_patients,
# MAGIC   COUNT(DISTINCT room_id) as rooms_used,
# MAGIC   AVG(TIMESTAMPDIFF(MINUTE, actual_start, actual_end)) as avg_duration_minutes,
# MAGIC   SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
# MAGIC   SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) as cancelled
# MAGIC FROM lab_of_the_future.digital_twin.surgery_sessions
# MAGIC WHERE scheduled_start >= DATEADD(DAY, -30, current_date())
# MAGIC GROUP BY DATE(scheduled_start)
# MAGIC ORDER BY date DESC
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: React App Component
# MAGIC 
# MAGIC Below is the main React component that should be placed in `client/src/App.tsx`:

# COMMAND ----------

# MAGIC %md
# MAGIC ### `App.tsx`
# MAGIC 
# MAGIC ```typescript
# MAGIC import React, { useState } from 'react';
# MAGIC import { 
# MAGIC   AppLayout, 
# MAGIC   Card, 
# MAGIC   DataTable, 
# MAGIC   LineChart,
# MAGIC   BarChart,
# MAGIC   useAnalyticsQuery,
# MAGIC   Badge,
# MAGIC   Tabs,
# MAGIC   Tab,
# MAGIC   Select,
# MAGIC   Option
# MAGIC } from '@databricks/appkit';
# MAGIC 
# MAGIC // Type definitions from appKitTypes.d.ts (generated by npm run typegen)
# MAGIC import type { 
# MAGIC   PatientSummaryRow, 
# MAGIC   RoomStatusRow, 
# MAGIC   ActiveAlertsRow,
# MAGIC   PatientVitalsHistoryRow,
# MAGIC   OperationalMetricsRow
# MAGIC } from './appKitTypes';
# MAGIC 
# MAGIC function App() {
# MAGIC   const [selectedPatient, setSelectedPatient] = useState<string | null>(null);
# MAGIC   const [activeTab, setActiveTab] = useState('overview');
# MAGIC 
# MAGIC   // Fetch data using generated query keys
# MAGIC   const { data: patients, isLoading: patientsLoading } = 
# MAGIC     useAnalyticsQuery<PatientSummaryRow[]>({ queryKey: 'patientSummary' });
# MAGIC   
# MAGIC   const { data: roomStatus, isLoading: roomsLoading } = 
# MAGIC     useAnalyticsQuery<RoomStatusRow[]>({ queryKey: 'roomStatus' });
# MAGIC   
# MAGIC   const { data: alerts, isLoading: alertsLoading } = 
# MAGIC     useAnalyticsQuery<ActiveAlertsRow[]>({ queryKey: 'activeAlerts' });
# MAGIC   
# MAGIC   const { data: metrics, isLoading: metricsLoading } = 
# MAGIC     useAnalyticsQuery<OperationalMetricsRow[]>({ queryKey: 'operationalMetrics' });
# MAGIC 
# MAGIC   const { data: vitalsHistory } = useAnalyticsQuery<PatientVitalsHistoryRow[]>({
# MAGIC     queryKey: 'patientVitalsHistory',
# MAGIC     params: { patient_id: selectedPatient },
# MAGIC     enabled: !!selectedPatient
# MAGIC   });
# MAGIC 
# MAGIC   // Calculate KPIs
# MAGIC   const activeRooms = roomStatus?.filter(r => r.current_status === 'in_progress').length || 0;
# MAGIC   const criticalAlerts = alerts?.filter(a => a.severity === 'CRITICAL' && a.alert_status === 'active').length || 0;
# MAGIC   const todaySurgeries = metrics?.[0]?.total_surgeries || 0;
# MAGIC 
# MAGIC   const getSeverityColor = (severity: string) => {
# MAGIC     switch (severity) {
# MAGIC       case 'CRITICAL': return 'red';
# MAGIC       case 'WARNING': return 'yellow';
# MAGIC       default: return 'blue';
# MAGIC     }
# MAGIC   };
# MAGIC 
# MAGIC   const getStatusColor = (status: string) => {
# MAGIC     switch (status) {
# MAGIC       case 'in_progress': return 'green';
# MAGIC       case 'critical': return 'red';
# MAGIC       case 'warning': return 'yellow';
# MAGIC       default: return 'gray';
# MAGIC     }
# MAGIC   };
# MAGIC 
# MAGIC   return (
# MAGIC     <AppLayout title="Lab of the Future" subtitle="Healthcare Digital Twin Dashboard">
# MAGIC       {/* KPI Cards */}
# MAGIC       <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginBottom: '24px' }}>
# MAGIC         <Card>
# MAGIC           <h3>Active Surgeries</h3>
# MAGIC           <p style={{ fontSize: '2rem', fontWeight: 'bold' }}>{activeRooms}</p>
# MAGIC         </Card>
# MAGIC         <Card>
# MAGIC           <h3>Critical Alerts</h3>
# MAGIC           <p style={{ fontSize: '2rem', fontWeight: 'bold', color: criticalAlerts > 0 ? 'red' : 'green' }}>
# MAGIC             {criticalAlerts}
# MAGIC           </p>
# MAGIC         </Card>
# MAGIC         <Card>
# MAGIC           <h3>Today's Surgeries</h3>
# MAGIC           <p style={{ fontSize: '2rem', fontWeight: 'bold' }}>{todaySurgeries}</p>
# MAGIC         </Card>
# MAGIC         <Card>
# MAGIC           <h3>Total Patients</h3>
# MAGIC           <p style={{ fontSize: '2rem', fontWeight: 'bold' }}>{patients?.length || 0}</p>
# MAGIC         </Card>
# MAGIC       </div>
# MAGIC 
# MAGIC       {/* Main Content Tabs */}
# MAGIC       <Tabs value={activeTab} onChange={setActiveTab}>
# MAGIC         <Tab value="overview" label="Room Overview" />
# MAGIC         <Tab value="patients" label="Patients" />
# MAGIC         <Tab value="alerts" label="Alerts" />
# MAGIC         <Tab value="analytics" label="Analytics" />
# MAGIC       </Tabs>
# MAGIC 
# MAGIC       {activeTab === 'overview' && (
# MAGIC         <Card title="Surgery Room Status">
# MAGIC           <DataTable
# MAGIC             data={roomStatus || []}
# MAGIC             loading={roomsLoading}
# MAGIC             columns={[
# MAGIC               { key: 'room_name', header: 'Room' },
# MAGIC               { 
# MAGIC                 key: 'current_status', 
# MAGIC                 header: 'Status',
# MAGIC                 render: (row) => (
# MAGIC                   <Badge color={getStatusColor(row.current_status)}>
# MAGIC                     {row.current_status}
# MAGIC                   </Badge>
# MAGIC                 )
# MAGIC               },
# MAGIC               { key: 'patient_id', header: 'Patient' },
# MAGIC               { key: 'procedure_type', header: 'Procedure' },
# MAGIC               { key: 'lead_surgeon', header: 'Surgeon' },
# MAGIC               { 
# MAGIC                 key: 'patient_heart_rate', 
# MAGIC                 header: 'Heart Rate',
# MAGIC                 render: (row) => row.patient_heart_rate ? `${row.patient_heart_rate} BPM` : '-'
# MAGIC               },
# MAGIC               { 
# MAGIC                 key: 'patient_status', 
# MAGIC                 header: 'Vitals',
# MAGIC                 render: (row) => row.patient_status ? (
# MAGIC                   <Badge color={getStatusColor(row.patient_status)}>
# MAGIC                     {row.patient_status}
# MAGIC                   </Badge>
# MAGIC                 ) : '-'
# MAGIC               },
# MAGIC               { key: 'personnel_count', header: 'Staff' },
# MAGIC               { key: 'active_alert_count', header: 'Alerts' }
# MAGIC             ]}
# MAGIC           />
# MAGIC         </Card>
# MAGIC       )}
# MAGIC 
# MAGIC       {activeTab === 'patients' && (
# MAGIC         <>
# MAGIC           <Card title="Patient Selection">
# MAGIC             <Select 
# MAGIC               value={selectedPatient || ''} 
# MAGIC               onChange={(e) => setSelectedPatient(e.target.value)}
# MAGIC               placeholder="Select a patient..."
# MAGIC             >
# MAGIC               {patients?.map(p => (
# MAGIC                 <Option key={p.patient_id} value={p.patient_id}>
# MAGIC                   {p.patient_id} - {p.ekg_recordings} EKGs, {p.imaging_studies} Images
# MAGIC                 </Option>
# MAGIC               ))}
# MAGIC             </Select>
# MAGIC           </Card>
# MAGIC           
# MAGIC           {selectedPatient && vitalsHistory && (
# MAGIC             <Card title={`Heart Rate History - ${selectedPatient}`}>
# MAGIC               <LineChart
# MAGIC                 data={vitalsHistory}
# MAGIC                 xKey="recording_datetime"
# MAGIC                 yKeys={['heart_rate_bpm']}
# MAGIC                 height={300}
# MAGIC               />
# MAGIC             </Card>
# MAGIC           )}
# MAGIC           
# MAGIC           <Card title="Patient Summary">
# MAGIC             <DataTable
# MAGIC               data={patients || []}
# MAGIC               loading={patientsLoading}
# MAGIC               columns={[
# MAGIC                 { key: 'patient_id', header: 'Patient ID' },
# MAGIC                 { key: 'ekg_recordings', header: 'EKG Recordings' },
# MAGIC                 { key: 'avg_heart_rate', header: 'Avg Heart Rate' },
# MAGIC                 { key: 'min_heart_rate', header: 'Min HR' },
# MAGIC                 { key: 'max_heart_rate', header: 'Max HR' },
# MAGIC                 { key: 'imaging_studies', header: 'Imaging Studies' },
# MAGIC                 { key: 'surgeries', header: 'Surgeries' }
# MAGIC               ]}
# MAGIC               onRowClick={(row) => setSelectedPatient(row.patient_id)}
# MAGIC             />
# MAGIC           </Card>
# MAGIC         </>
# MAGIC       )}
# MAGIC 
# MAGIC       {activeTab === 'alerts' && (
# MAGIC         <Card title="Alert Dashboard">
# MAGIC           <DataTable
# MAGIC             data={alerts || []}
# MAGIC             loading={alertsLoading}
# MAGIC             columns={[
# MAGIC               { 
# MAGIC                 key: 'severity', 
# MAGIC                 header: 'Severity',
# MAGIC                 render: (row) => (
# MAGIC                   <Badge color={getSeverityColor(row.severity)}>
# MAGIC                     {row.severity}
# MAGIC                   </Badge>
# MAGIC                 )
# MAGIC               },
# MAGIC               { key: 'room_name', header: 'Room' },
# MAGIC               { key: 'patient_id', header: 'Patient' },
# MAGIC               { key: 'message', header: 'Message' },
# MAGIC               { 
# MAGIC                 key: 'alert_status', 
# MAGIC                 header: 'Status',
# MAGIC                 render: (row) => (
# MAGIC                   <Badge color={row.alert_status === 'active' ? 'red' : 'gray'}>
# MAGIC                     {row.alert_status}
# MAGIC                   </Badge>
# MAGIC                 )
# MAGIC               },
# MAGIC               { key: 'triggered_at', header: 'Triggered' },
# MAGIC               { key: 'duration_minutes', header: 'Duration (min)' }
# MAGIC             ]}
# MAGIC           />
# MAGIC         </Card>
# MAGIC       )}
# MAGIC 
# MAGIC       {activeTab === 'analytics' && (
# MAGIC         <>
# MAGIC           <Card title="Daily Surgery Volume (Last 30 Days)">
# MAGIC             <BarChart
# MAGIC               data={metrics || []}
# MAGIC               xKey="date"
# MAGIC               yKeys={['total_surgeries', 'completed', 'cancelled']}
# MAGIC               height={300}
# MAGIC             />
# MAGIC           </Card>
# MAGIC           <Card title="Operational Metrics">
# MAGIC             <DataTable
# MAGIC               data={metrics || []}
# MAGIC               loading={metricsLoading}
# MAGIC               columns={[
# MAGIC                 { key: 'date', header: 'Date' },
# MAGIC                 { key: 'total_surgeries', header: 'Total Surgeries' },
# MAGIC                 { key: 'unique_patients', header: 'Patients' },
# MAGIC                 { key: 'rooms_used', header: 'Rooms Used' },
# MAGIC                 { key: 'avg_duration_minutes', header: 'Avg Duration (min)' },
# MAGIC                 { key: 'completed', header: 'Completed' },
# MAGIC                 { key: 'cancelled', header: 'Cancelled' }
# MAGIC               ]}
# MAGIC             />
# MAGIC           </Card>
# MAGIC         </>
# MAGIC       )}
# MAGIC     </AppLayout>
# MAGIC   );
# MAGIC }
# MAGIC 
# MAGIC export default App;
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Deploy the Application
# MAGIC 
# MAGIC After setting up the app structure, deploy using these commands:
# MAGIC 
# MAGIC ```bash
# MAGIC # Navigate to the app directory
# MAGIC cd lab-of-future
# MAGIC 
# MAGIC # Install dependencies
# MAGIC npm install
# MAGIC 
# MAGIC # Generate TypeScript types from SQL queries
# MAGIC npm run typegen
# MAGIC 
# MAGIC # Validate the app configuration
# MAGIC databricks apps validate --profile <your-profile>
# MAGIC 
# MAGIC # Deploy the app
# MAGIC databricks apps deploy --profile <your-profile>
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Alternative - Quick Dashboard with Streamlit
# MAGIC 
# MAGIC For rapid prototyping, you can also create a Streamlit app directly in Databricks:

# COMMAND ----------

# Save Streamlit app code (for reference - run in separate notebook or as app)
streamlit_code = '''
import streamlit as st
import pandas as pd
from databricks import sql
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Lab of the Future",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Lab of the Future Dashboard")
st.markdown("### Healthcare Digital Twin Monitoring System")

# Connection helper (requires databricks-sql-connector)
@st.cache_resource
def get_connection():
    return sql.connect(
        server_hostname=st.secrets["DATABRICKS_HOST"],
        http_path=st.secrets["DATABRICKS_HTTP_PATH"],
        access_token=st.secrets["DATABRICKS_TOKEN"]
    )

@st.cache_data(ttl=60)
def run_query(query):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(query)
    columns = [desc[0] for desc in cursor.description]
    data = cursor.fetchall()
    return pd.DataFrame(data, columns=columns)

# Sidebar filters
st.sidebar.header("Filters")
refresh = st.sidebar.button("🔄 Refresh Data")

# KPI Row
col1, col2, col3, col4 = st.columns(4)

room_status = run_query("""
    SELECT * FROM lab_of_the_future.digital_twin.app_room_status
""")

alerts = run_query("""
    SELECT * FROM lab_of_the_future.digital_twin.app_alerts_dashboard
    WHERE alert_status != 'resolved'
""")

active_rooms = len(room_status[room_status['current_status'] == 'in_progress'])
critical_alerts = len(alerts[alerts['severity'] == 'CRITICAL'])

with col1:
    st.metric("Active Surgeries", active_rooms)
with col2:
    st.metric("Critical Alerts", critical_alerts, delta_color="inverse")
with col3:
    st.metric("Total Rooms", len(room_status))
with col4:
    patients = run_query("SELECT COUNT(DISTINCT patient_id) as cnt FROM lab_of_the_future.healthcare_data.app_patient_summary")
    st.metric("Total Patients", patients['cnt'].iloc[0])

# Tabs
tab1, tab2, tab3 = st.tabs(["🏠 Room Status", "⚠️ Alerts", "📊 Analytics"])

with tab1:
    st.subheader("Surgery Room Status")
    
    # Color code status
    def status_color(status):
        if status == 'in_progress':
            return '🟢'
        elif status == 'critical':
            return '🔴'
        elif status == 'warning':
            return '🟡'
        return '⚪'
    
    room_status['Status Icon'] = room_status['current_status'].apply(status_color)
    st.dataframe(
        room_status[['room_name', 'Status Icon', 'current_status', 'patient_id', 
                     'procedure_type', 'lead_surgeon', 'patient_heart_rate', 'patient_status']],
        use_container_width=True
    )

with tab2:
    st.subheader("Alert Dashboard")
    
    # Filter by severity
    severity_filter = st.multiselect(
        "Filter by Severity",
        options=['CRITICAL', 'WARNING', 'INFO'],
        default=['CRITICAL', 'WARNING']
    )
    
    filtered_alerts = alerts[alerts['severity'].isin(severity_filter)]
    st.dataframe(filtered_alerts, use_container_width=True)

with tab3:
    st.subheader("Operational Analytics")
    
    metrics = run_query("""
        SELECT * FROM lab_of_the_future.digital_twin.surgery_sessions
        WHERE scheduled_start >= DATEADD(DAY, -30, current_date())
    """)
    
    if not metrics.empty:
        metrics['date'] = pd.to_datetime(metrics['scheduled_start']).dt.date
        daily = metrics.groupby('date').size().reset_index(name='surgeries')
        
        fig = px.bar(daily, x='date', y='surgeries', title='Daily Surgery Volume')
        st.plotly_chart(fig, use_container_width=True)
'''

print("Streamlit app code generated for reference")
print("To use: Create a new Streamlit app in Databricks Apps and paste the code above")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Application Screenshots (Expected UI)
# MAGIC 
# MAGIC The deployed application will feature:
# MAGIC 
# MAGIC 1. **Dashboard Header**: KPI cards showing active surgeries, alerts, and patient counts
# MAGIC 2. **Room Overview Tab**: Real-time status of all surgery rooms with vital signs
# MAGIC 3. **Patients Tab**: Patient list with EKG history charts
# MAGIC 4. **Alerts Tab**: Filterable alert dashboard with severity indicators
# MAGIC 5. **Analytics Tab**: Historical trends and operational metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## Security Considerations
# MAGIC 
# MAGIC When deploying healthcare applications:
# MAGIC 
# MAGIC 1. **Data Access**: Use Unity Catalog permissions to restrict access to PHI
# MAGIC 2. **Authentication**: Leverage Databricks workspace authentication
# MAGIC 3. **Audit Logging**: Enable audit logs for compliance (HIPAA, etc.)
# MAGIC 4. **Data Masking**: Consider masking patient identifiers in non-production environments
# MAGIC 
# MAGIC ```sql
# MAGIC -- Example: Grant read access to the app service principal
# MAGIC GRANT SELECT ON lab_of_the_future.healthcare_data.app_patient_summary TO `app-service-principal`;
# MAGIC GRANT SELECT ON lab_of_the_future.digital_twin.app_room_status TO `app-service-principal`;
# MAGIC GRANT SELECT ON lab_of_the_future.digital_twin.app_alerts_dashboard TO `app-service-principal`;
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC Continue to **Notebook 5** to learn how to:
# MAGIC - Set up Delta Sharing for DICOM image volumes
# MAGIC - Share featurized data with external partners
# MAGIC - Configure secure data sharing for collaborative research
