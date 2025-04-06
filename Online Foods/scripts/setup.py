import subprocess
import sys
import os

def check_python_version():
    """Check if Python version is at least 3.8."""
    required_version = (3, 8)
    current_version = sys.version_info
    
    if current_version < required_version:
        print(f"Error: Python {required_version[0]}.{required_version[1]} or higher is required.")
        print(f"Current Python version: {current_version[0]}.{current_version[1]}.{current_version[2]}")
        sys.exit(1)
    
    print(f"✅ Python version check passed: {current_version[0]}.{current_version[1]}.{current_version[2]}")

def install_dependencies():
    """Install dependencies from requirements.txt."""
    print("\nInstalling dependencies from requirements.txt...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        sys.exit(1)

def check_csv_file():
    """Check if the CSV file exists."""
    if not os.path.exists('data/onlinefoods.csv'):
        print("❌ Error: data/onlinefoods.csv file not found!")
        print("Please make sure the CSV file is in the data directory.")
        sys.exit(1)
    
    print("✅ CSV file check passed: data/onlinefoods.csv found!")

def create_directories():
    """Create necessary directories."""
    directories = ['output/visualizations', 'output/models', 'output/dashboards', 'output/quick_start_viz']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"✅ Directory already exists: {directory}")

def main():
    """Main function to set up the environment."""
    print("=" * 80)
    print("Setting up the Online Foods EDA environment")
    print("=" * 80)
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    install_dependencies()
    
    # Check CSV file
    check_csv_file()
    
    # Create directories
    create_directories()
    
    print("\n" + "=" * 80)
    print("Setup completed successfully!")
    print("You can now run the analysis with: python scripts/run_analysis.py")
    print("=" * 80)

if __name__ == "__main__":
    main() 