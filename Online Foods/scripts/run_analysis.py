import os
import subprocess
import time
import webbrowser
from threading import Timer

def run_command(command, description):
    """Run a command and print its description."""
    print(f"\n{'=' * 80}")
    print(f"Running: {description}")
    print(f"{'=' * 80}\n")
    
    process = subprocess.Popen(command, shell=True)
    process.wait()
    
    if process.returncode == 0:
        print(f"\n✅ {description} completed successfully!")
    else:
        print(f"\n❌ {description} failed with return code {process.returncode}")
        exit(1)

def open_browser(url):
    """Open a browser to the specified URL."""
    webbrowser.open(url)

def main():
    # Check if the CSV file exists
    if not os.path.exists('data/onlinefoods.csv'):
        print("Error: data/onlinefoods.csv file not found!")
        exit(1)
    
    # Create output directories if they don't exist
    for directory in ['output/visualizations', 'output/models', 'output/dashboards']:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Run the basic EDA script
    run_command('python scripts/online_foods_eda.py', 'Basic Exploratory Data Analysis')
    
    # Run the machine learning script
    run_command('python scripts/online_foods_ml.py', 'Machine Learning Analysis')
    
    # Start the dashboard
    print("\n{'=' * 80}")
    print("Starting the interactive dashboard...")
    print(f"{'=' * 80}\n")
    
    # Open browser after a delay to ensure the server is running
    Timer(3, open_browser, args=['http://127.0.0.1:8050']).start()
    
    # Run the dashboard
    subprocess.run('python scripts/online_foods_dashboard.py', shell=True)

if __name__ == "__main__":
    # Start the analysis
    start_time = time.time()
    main()
    end_time = time.time()
    
    # Print execution time
    execution_time = end_time - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)") 