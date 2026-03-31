"""
Lab of the Future - Streamlit Dashboard
Alternative implementation using Streamlit for rapid prototyping.

To deploy as a Databricks App:
1. Create a new Streamlit app: databricks apps init --name lab-future-streamlit --features streamlit
2. Copy this file to the app directory
3. Deploy: databricks apps deploy
"""

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
