#!/usr/bin/env python3
"""
Setup script for the Modern Emotion Recognition System.
"""

import subprocess
import sys
from pathlib import Path
import argparse


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True


def install_dependencies() -> bool:
    """Install project dependencies."""
    return run_command("pip install -r requirements.txt", "Installing dependencies")


def create_directories() -> bool:
    """Create necessary directories."""
    directories = ["data", "models", "logs", "reports"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True


def run_tests() -> bool:
    """Run the test suite."""
    return run_command("python -m pytest tests/ -v", "Running tests")


def create_sample_data() -> bool:
    """Create sample data for testing."""
    try:
        from src.emotion_detector import create_synthetic_dataset
        data_dir = Path("data")
        create_synthetic_dataset(data_dir, 50)
        print("✅ Sample data created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")
        return False


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Modern Emotion Recognition System")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--skip-data", action="store_true", help="Skip creating sample data")
    parser.add_argument("--dev", action="store_true", help="Install development dependencies")
    
    args = parser.parse_args()
    
    print("🚀 Setting up Modern Emotion Recognition System...")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("❌ Setup failed during directory creation")
        sys.exit(1)
    
    # Create sample data
    if not args.skip_data:
        if not create_sample_data():
            print("⚠️  Sample data creation failed, but continuing...")
    
    # Run tests
    if not args.skip_tests:
        if not run_tests():
            print("⚠️  Tests failed, but setup completed")
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Launch the web interface: streamlit run web_app/app.py")
    print("2. Or use the CLI: python src/cli.py --text 'Hello world!'")
    print("3. Check the README.md for more examples")
    print("\n🔗 Useful commands:")
    print("- Web interface: streamlit run web_app/app.py")
    print("- CLI help: python src/cli.py --help")
    print("- Run tests: python -m pytest tests/ -v")
    print("- Create sample data: python src/cli.py --create-sample sample.json")


if __name__ == "__main__":
    main()
