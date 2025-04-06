import subprocess
import sys
import os
import platform

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
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("✅ Pip upgraded successfully!")
        
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        sys.exit(1)

def install_additional_system_dependencies():
    """Install system-specific dependencies."""
    system = platform.system().lower()
    
    if system == "linux":
        try:
            print("\nInstalling additional system dependencies for Linux...")
            # For Ubuntu/Debian-based systems
            subprocess.check_call(["apt-get", "update"])
            subprocess.check_call(["apt-get", "install", "-y", "python3-tk"])
            print("✅ System dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Warning: Could not install system dependencies: {e}")
            print("You may need to manually install tkinter for your Linux distribution.")
    elif system == "darwin":  # macOS
        print("\nFor macOS users:")
        print("If you encounter issues with matplotlib, you may need to install tkinter:")
        print("brew install python-tk")
    elif system == "windows":
        print("\nFor Windows users:")
        print("If you encounter issues with matplotlib, ensure you have the Microsoft Visual C++ Redistributable installed.")

def main():
    """Main function to install dependencies."""
    print("=" * 80)
    print("Installing dependencies for Online Foods EDA")
    print("=" * 80)
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    install_dependencies()
    
    # Install system-specific dependencies
    install_additional_system_dependencies()
    
    print("\n" + "=" * 80)
    print("Installation completed successfully!")
    print("You can now run the analysis with: python quick_start.py")
    print("=" * 80)

if __name__ == "__main__":
    main() 