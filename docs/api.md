# DrishtiXAI REST API Specification

## Endpoints

### 1. Analyze Retinal Image
`POST /api/analyze`
- **Request**: `multipart/form-data` with `image` file and `case_id`, `metadata`.
- **Response**: Canonical Analysis JSON matching `DEMO_CASES[*].result`.

### 2. Cases Management
`GET /api/cases`
`GET /api/cases/{case_id}`
`POST /api/cases`

### 3. Reports
`GET /api/reports/{screening_id}`
`GET /api/reports/{screening_id}/pdf`

### 4. Specialist Review
`POST /api/review/{screening_id}`

### 5. System Status
`GET /api/system/status`
`GET /api/health`
