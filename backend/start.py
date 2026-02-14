"""
EssentiaX Backend Startup Script
"""
import os
import sys
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import sqlalchemy
        import pandas
        import uvicorn
        print("✅ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def setup_database():
    """Setup database and run migrations"""
    print("🗄️ Setting up database...")
    
    # Check if alembic is available
    try:
        result = subprocess.run(["alembic", "current"], capture_output=True, text=True)
        if result.returncode != 0:
            print("📝 Initializing database migrations...")
            subprocess.run(["alembic", "revision", "--autogenerate", "-m", "Initial migration"], check=True)
        
        print("🔄 Running database migrations...")
        subprocess.run(["alembic", "upgrade", "head"], check=True)
        print("✅ Database setup complete")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Database setup failed: {e}")
        return False
    except FileNotFoundError:
        print("⚠️ Alembic not found, skipping migrations")
        print("Database tables will be created automatically on first run")
        return True

def create_directories():
    """Create necessary directories"""
    directories = ["uploads", "reports", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")

def check_environment():
    """Check environment configuration"""
    print("🔧 Checking environment configuration...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️ .env file not found, using default configuration")
        print("💡 Copy .env.example to .env and customize for production")
    else:
        print("✅ Environment file found")
    
    # Check critical environment variables
    from app.core.config import settings
    
    if settings.SECRET_KEY == "your-secret-key-change-in-production":
        print("⚠️ WARNING: Using default SECRET_KEY - change this in production!")
    
    print(f"📊 Database URL: {settings.DATABASE_URL}")
    print(f"🔗 Redis URL: {settings.REDIS_URL}")
    print(f"📁 Upload directory: {settings.UPLOAD_DIR}")
    print(f"📄 Reports directory: {settings.REPORTS_DIR}")

def run_tests():
    """Run backend tests"""
    print("🧪 Running backend tests...")
    try:
        from test_backend import run_tests
        success = run_tests()
        if success:
            print("✅ All tests passed")
            return True
        else:
            print("❌ Some tests failed")
            return False
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

def start_server(host="0.0.0.0", port=8000, reload=True):
    """Start the FastAPI server"""
    print(f"🚀 Starting EssentiaX Backend server on {host}:{port}")
    print(f"📖 API Documentation: http://{host}:{port}/docs")
    print(f"🔍 Health Check: http://{host}:{port}/health")
    print("Press Ctrl+C to stop the server")
    
    try:
        import uvicorn
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")

def main():
    """Main startup function"""
    print("🧠 EssentiaX Backend - Phase 2 Implementation")
    print("=" * 50)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "test":
            if check_dependencies():
                run_tests()
            return
        
        elif command == "setup":
            print("🔧 Setting up EssentiaX Backend...")
            if check_dependencies():
                create_directories()
                check_environment()
                setup_database()
                print("✅ Setup complete!")
            return
        
        elif command == "dev":
            # Development mode with auto-reload
            if check_dependencies():
                create_directories()
                check_environment()
                start_server(reload=True)
            return
        
        elif command == "prod":
            # Production mode
            if check_dependencies():
                create_directories()
                check_environment()
                setup_database()
                start_server(reload=False)
            return
        
        else:
            print(f"Unknown command: {command}")
            print("Available commands: setup, test, dev, prod")
            return
    
    # Default: interactive setup and start
    print("🚀 Starting EssentiaX Backend...")
    
    # Step 1: Check dependencies
    if not check_dependencies():
        return
    
    # Step 2: Create directories
    create_directories()
    
    # Step 3: Check environment
    check_environment()
    
    # Step 4: Setup database
    setup_database()
    
    # Step 5: Ask user what to do
    print("\n" + "=" * 50)
    print("What would you like to do?")
    print("1. Run tests")
    print("2. Start development server")
    print("3. Start production server")
    print("4. Exit")
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "1":
        run_tests()
    elif choice == "2":
        start_server(reload=True)
    elif choice == "3":
        start_server(reload=False)
    elif choice == "4":
        print("👋 Goodbye!")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()