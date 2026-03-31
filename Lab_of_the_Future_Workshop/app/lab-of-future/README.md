# Lab of the Future - Healthcare Dashboard

A Databricks App for monitoring healthcare digital twin data including EKG vitals, medical imaging, and surgery room operations.

## Features

- **Room Overview**: Real-time status of all surgery rooms with vital signs
- **Patient Dashboard**: Patient list with EKG history charts
- **Alert Management**: Filterable alert dashboard with severity indicators
- **Analytics**: Historical trends and operational metrics

## Prerequisites

1. Complete Notebooks 1-3 to populate the data tables
2. Databricks CLI installed and authenticated
3. Access to a SQL Warehouse

## Directory Structure

```
lab-of-future/
├── app.yaml                    # App configuration
├── package.json                # Node.js dependencies
├── tsconfig.json               # TypeScript configuration
├── vite.config.ts              # Vite build configuration
├── config/
│   └── queries/                # SQL queries for analytics
│       ├── patientSummary.sql
│       ├── roomStatus.sql
│       ├── activeAlerts.sql
│       ├── patientVitalsHistory.sql
│       ├── imagingHistory.sql
│       └── operationalMetrics.sql
└── client/
    ├── index.html
    └── src/
        ├── main.tsx            # React entry point
        └── App.tsx             # Main application component
```

## Deployment

### 1. Configure Warehouse ID

Edit `app.yaml` and replace `${DATABRICKS_WAREHOUSE_ID}` with your SQL Warehouse ID, or set it as an environment variable.

To find your warehouse ID:
```bash
databricks sql warehouses list --profile <your-profile>
```

### 2. Install Dependencies

```bash
cd lab-of-future
npm install
```

### 3. Generate TypeScript Types

```bash
npm run typegen
```

This generates `client/src/appKitTypes.d.ts` with type definitions for your SQL queries.

### 4. Validate Configuration

```bash
databricks apps validate --profile <your-profile>
```

### 5. Deploy

```bash
databricks apps deploy --profile <your-profile>
```

## Local Development

```bash
npm run dev
```

Note: Local development requires connection to Databricks for data queries.

## Required Database Views

Ensure these views exist (created by Notebook 4):

- `lab_of_the_future.healthcare_data.app_patient_summary`
- `lab_of_the_future.digital_twin.app_room_status`
- `lab_of_the_future.digital_twin.app_alerts_dashboard`

## Security

Grant appropriate permissions to the app service principal:

```sql
GRANT SELECT ON lab_of_the_future.healthcare_data.app_patient_summary TO `app-service-principal`;
GRANT SELECT ON lab_of_the_future.digital_twin.app_room_status TO `app-service-principal`;
GRANT SELECT ON lab_of_the_future.digital_twin.app_alerts_dashboard TO `app-service-principal`;
GRANT SELECT ON lab_of_the_future.healthcare_data.ekg_silver TO `app-service-principal`;
GRANT SELECT ON lab_of_the_future.healthcare_data.dicom_metadata TO `app-service-principal`;
GRANT SELECT ON lab_of_the_future.digital_twin.surgery_sessions TO `app-service-principal`;
```
