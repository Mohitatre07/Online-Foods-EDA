#!/bin/bash

echo "==================================================="
echo "Online Foods Exploratory Data Analysis"
echo "==================================================="
echo

echo "Step 1: Installing dependencies..."
python scripts/install_dependencies.py
if [ $? -ne 0 ]; then
    echo "Error installing dependencies!"
    read -p "Press Enter to continue..."
    exit 1
fi

echo
echo "Step 2: Setting up the environment..."
python scripts/setup.py
if [ $? -ne 0 ]; then
    echo "Error setting up the environment!"
    read -p "Press Enter to continue..."
    exit 1
fi

echo
echo "Step 3: Running the analysis..."
python scripts/run_analysis.py
if [ $? -ne 0 ]; then
    echo "Error running the analysis!"
    read -p "Press Enter to continue..."
    exit 1
fi

echo
echo "==================================================="
echo "Analysis completed successfully!"
echo "==================================================="
echo
echo "Check the output directory for results."
echo
read -p "Press Enter to continue..." 