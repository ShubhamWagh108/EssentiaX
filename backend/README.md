# EssentiaX Backend - Phase 2 Implementation

## 🚀 Overview

The EssentiaX Backend is a FastAPI-based REST API that provides comprehensive data analysis capabilities with user authentication, dataset management, and report generation. This is the Phase 2 implementation that adds backend integration to the unified EDA engine from Phase 1.

## ✨ Features

### 🔐 Authentication & User Management
- JWT-based authentication with access and refresh tokens
- User registration and login
- Profile management
- Role-based access control

### 📊 Dataset Management
- File upload support (CSV, Excel, JSON)
- Dataset metadata extraction
- File validation and security
- Dataset preview functionality
- Public/private dataset sharing

### 📈 Report Generation
- Automated EDA report generation using the unified smart_eda engine
- Multiple output formats (HTML, JSON, Interactive)
- Background processing for large datasets
- Report sharing with secure tokens
- Dashboard with analytics

### 🗄️ Database Integration
- PostgreSQL for production data storage
- SQLite for development and testing
- Alembic for database migrations
- Comprehensive data models

### 🔧 Additional Features
- Redis for caching and sessions
- Background task processing with Celery
- File storage management
- CORS support for frontend integration
- Comprehensive API documentation
- Health checks and monitoring

## 🏗️ Architecture

```
backend/
├── app/
│   ├── api/                 # API endpoints
│   │   └── v1/             # API version 1
│   │       ├── auth.py     # Authentication endpoints
│   │       ├── users.py    # User management
│   │       ├── datasets.py # Dataset operations
│   │       └── reports.py  # Report generation
│   ├── core/               # Core functionality
│   │   ├── config.py       # Configuration settings
│   │   └── security.py     # Security utilities
│   ├── crud/               # Database operations
│   │   ├── user.py         # User CRUD
│   │   └── report.py       # Report/Dataset CRUD
│   ├── db/                 # Database configuration
│   │   └── database.py     # SQLAlchemy setup
│   ├── models/             # Database models
│   │   ├── user.py         # User model
│   │   └── report.py       # Report/Dataset models
│   ├── schemas/            # Pydantic schemas
│   │   ├── user.py         # User schemas
│   │   └── report.py       # Report schemas
│   └── main.py             # FastAPI application
├── alembic/                # Database migrations
├── uploads/                # Uploaded datasets
├── reports/                # Generated reports
└── tests/                  # Test files
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL (for production)
- Redis (for caching)

### Installation

1. **Clone and navigate to backend directory:**
```bash
cd backend
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Setup environment:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Start with the setup script:**
```bash
python start.py setup
```

### Development Mode

**Quick start:**
```bash
python start.py dev
```

**Manual start:**
```bash
# Setup database
alembic upgrade head

# Start development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

**Using Docker Compose (Recommended):**
```bash
docker-compose up -d
```

**Manual production setup:**
```bash
python start.py prod
```

## 🧪 Testing

**Run all tests:**
```bash
python start.py test
```

**Run specific tests:**
```bash
python test_backend.py
```

**Test coverage includes:**
- Authentication and authorization
- User management
- Dataset upload and processing
- Report generation
- API endpoints
- Error handling

## 📚 API Documentation

Once the server is running, visit:
- **Interactive API Docs:** http://localhost:8000/docs
- **ReDoc Documentation:** http://localhost:8000/redoc
- **OpenAPI Schema:** http://localhost:8000/api/v1/openapi.json

## 🔗 API Endpoints

### Authentication
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login (OAuth2)
- `POST /api/v1/auth/login/json` - JSON login
- `POST /api/v1/auth/refresh` - Refresh access token

### Users
- `GET /api/v1/users/me` - Get current user profile
- `PUT /api/v1/users/me` - Update user profile
- `GET /api/v1/users/{user_id}` - Get user by ID (admin)

### Datasets
- `POST /api/v1/datasets/upload` - Upload dataset file
- `GET /api/v1/datasets/` - List user datasets
- `GET /api/v1/datasets/{id}` - Get dataset details
- `GET /api/v1/datasets/{id}/preview` - Preview dataset
- `PUT /api/v1/datasets/{id}` - Update dataset
- `DELETE /api/v1/datasets/{id}` - Delete dataset

### Reports
- `POST /api/v1/reports/analyze` - Create EDA analysis
- `GET /api/v1/reports/` - List user reports
- `GET /api/v1/reports/public` - List public reports
- `GET /api/v1/reports/{id}` - Get report details
- `GET /api/v1/reports/{id}/download` - Download HTML report
- `GET /api/v1/reports/{id}/results` - Get JSON results
- `POST /api/v1/reports/{id}/share` - Share report
- `GET /api/v1/reports/shared/{token}` - Access shared report
- `GET /api/v1/reports/dashboard/stats` - Dashboard statistics

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Application
APP_NAME=EssentiaX Backend
DEBUG=false

# Security
SECRET_KEY=your-super-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Database
DATABASE_URL=postgresql://user:pass@localhost/db

# Redis
REDIS_URL=redis://localhost:6379

# File Storage
UPLOAD_DIR=uploads
REPORTS_DIR=reports
MAX_FILE_SIZE=104857600  # 100MB

# CORS
BACKEND_CORS_ORIGINS=http://localhost:3000,http://localhost:8000
```

### Database Configuration

**PostgreSQL (Production):**
```bash
DATABASE_URL=postgresql://essentiax:password@localhost/essentiax_db
```

**SQLite (Development):**
```bash
DATABASE_URL=sqlite:///./essentiax.db
```

## 🐳 Docker Deployment

**Start all services:**
```bash
docker-compose up -d
```

**Services included:**
- `api` - FastAPI backend server
- `postgres` - PostgreSQL database
- `redis` - Redis cache
- `celery` - Background task worker

**View logs:**
```bash
docker-compose logs -f api
```

## 🔄 Database Migrations

**Create new migration:**
```bash
alembic revision --autogenerate -m "Description"
```

**Apply migrations:**
```bash
alembic upgrade head
```

**Rollback migration:**
```bash
alembic downgrade -1
```

## 📊 Usage Examples

### 1. User Registration and Login

```python
import requests

# Register user
response = requests.post("http://localhost:8000/api/v1/auth/register", json={
    "email": "user@example.com",
    "username": "testuser",
    "password": "securepassword123",
    "full_name": "Test User"
})

# Login
response = requests.post("http://localhost:8000/api/v1/auth/login/json", json={
    "email": "user@example.com",
    "password": "securepassword123"
})

token = response.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}
```

### 2. Dataset Upload

```python
# Upload CSV file
with open("dataset.csv", "rb") as f:
    files = {"file": ("dataset.csv", f, "text/csv")}
    data = {
        "name": "My Dataset",
        "description": "Sample dataset for analysis",
        "is_public": False
    }
    response = requests.post(
        "http://localhost:8000/api/v1/datasets/upload",
        files=files,
        data=data,
        headers=headers
    )

dataset_id = response.json()["dataset_id"]
```

### 3. Generate EDA Report

```python
# Create analysis request
analysis_request = {
    "dataset_id": dataset_id,
    "report_title": "Comprehensive EDA Report",
    "report_description": "Full analysis of the dataset",
    "target_column": "target",
    "analysis_mode": "all",
    "sample_size": 10000,
    "max_plots": 8
}

response = requests.post(
    "http://localhost:8000/api/v1/reports/analyze",
    json=analysis_request,
    headers=headers
)

report_id = response.json()["report_id"]
```

### 4. Download Report

```python
# Download HTML report
response = requests.get(
    f"http://localhost:8000/api/v1/reports/{report_id}/download",
    headers=headers
)

with open("report.html", "wb") as f:
    f.write(response.content)
```

## 🔒 Security Features

- **JWT Authentication:** Secure token-based authentication
- **Password Hashing:** Bcrypt for secure password storage
- **File Validation:** Type and size validation for uploads
- **CORS Protection:** Configurable cross-origin policies
- **SQL Injection Protection:** SQLAlchemy ORM prevents SQL injection
- **Rate Limiting:** Built-in request rate limiting
- **Secure Headers:** Security headers for API responses

## 📈 Performance Features

- **Background Processing:** Celery for long-running tasks
- **Caching:** Redis for session and data caching
- **Database Optimization:** Connection pooling and query optimization
- **File Streaming:** Efficient file upload/download handling
- **Pagination:** Built-in pagination for large datasets

## 🐛 Troubleshooting

### Common Issues

**1. Database Connection Error:**
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Check connection string in .env
DATABASE_URL=postgresql://user:pass@localhost/db
```

**2. Redis Connection Error:**
```bash
# Check Redis is running
redis-cli ping

# Should return PONG
```

**3. File Upload Issues:**
```bash
# Check upload directory permissions
chmod 755 uploads/

# Check file size limits in .env
MAX_FILE_SIZE=104857600
```

**4. Migration Issues:**
```bash
# Reset migrations (development only)
alembic downgrade base
alembic upgrade head
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions:
- Check the API documentation at `/docs`
- Review the test files for usage examples
- Open an issue on GitHub

---

**Phase 2 Backend Integration Complete! 🎉**

The EssentiaX Backend provides a robust, scalable foundation for data analysis applications with comprehensive authentication, dataset management, and report generation capabilities.