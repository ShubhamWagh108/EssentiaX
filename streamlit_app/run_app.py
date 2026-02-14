"""
EssentiaX Streamlit App Launcher
===============================
Simple script to launch the EssentiaX Streamlit dashboard
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'plotly',
        'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("📦 Install them with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All required dependencies are installed")
    return True

def check_essentiax():
    """Check if EssentiaX modules are available"""
    try:
        sys.path.append(str(Path(__file__).parent.parent))
        from essentiax.io.smart_read import smart_read
        from essentiax.eda.smart_eda import smart_eda
        print("✅ EssentiaX modules are available")
        return True
    except ImportError as e:
        print(f"⚠️ EssentiaX modules not fully available: {e}")
        print("📝 The app will run with limited functionality")
        return False

def launch_app():
    """Launch the Streamlit app"""
    print("🚀 Launching EssentiaX Streamlit Dashboard...")
    print("📱 The app will open in your default web browser")
    print("🔗 URL: http://localhost:8501")
    print("\n" + "="*50)
    
    try:
        # Change to the streamlit_app directory
        app_dir = Path(__file__).parent
        os.chdir(app_dir)
        
        # Launch Streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'main.py',
            '--server.port', '8501',
            '--server.address', 'localhost',
            '--browser.gatherUsageStats', 'false'
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 App stopped by user")
    except Exception as e:
        print(f"❌ Error launching app: {e}")

def main():
    """Main launcher function"""
    print("🧠 EssentiaX Streamlit Dashboard Launcher")
    print("=" * 45)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check EssentiaX availability
    check_essentiax()
    
    print("\n🎯 Starting the dashboard...")
    print("💡 Tip: Press Ctrl+C to stop the app")
    
    # Launch the app
    launch_app()

if __name__ == "__main__":
    main()