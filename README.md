# Online Foods Exploratory Data Analysis

This project performs a comprehensive exploratory data analysis (EDA) on the Online Foods dataset to understand patterns and relationships in online food service usage.

## Dataset

The dataset contains information about individuals and their online food service usage, including:

- Demographic information (Age, Gender, Marital Status)
- Socioeconomic factors (Occupation, Monthly Income, Educational Qualifications)
- Household information (Family size)
- Geographic data (latitude, longitude, Pin code)
- Online food service usage (Output)
- Feedback on the service

## Project Structure

```
.
├── data/                        # Data directory
│   └── onlinefoods.csv          # Dataset file
├── scripts/                     # Scripts directory
│   ├── install_dependencies.py  # Dependency installation script
│   ├── setup.py                 # Setup script
│   ├── quick_start.py           # Quick overview script
│   ├── run_analysis.py          # Main analysis runner
│   ├── online_foods_eda.py      # Basic EDA script
│   ├── online_foods_dashboard.py # Interactive dashboard
│   └── online_foods_ml.py       # Machine learning analysis
├── output/                      # Output directory
│   ├── visualizations/          # Generated visualizations (PNG)
│   ├── models/                  # Trained ML models (PKL)
│   ├── dashboards/              # Interactive dashboards (HTML)
│   └── quick_start_viz/         # Quick start visualizations
├── README.md                    # Project documentation
├── requirements.txt             # Dependencies
├── run_analysis.bat             # Windows batch script
└── run_analysis.sh              # Linux/macOS shell script
```

## Getting Started

### One-Click Execution

For the easiest way to run the complete analysis:

- **Windows users**: Double-click on `run_analysis.bat`
- **Linux/macOS users**: Run `./run_analysis.sh` (you may need to make it executable first with `chmod +x run_analysis.sh`)

These scripts will automatically:
1. Install all dependencies
2. Set up the environment
3. Run the complete analysis

### Installation

To install all required dependencies manually:

```
python scripts/install_dependencies.py
```

This script will:
- Check if you have the required Python version (3.8+)
- Upgrade pip to the latest version
- Install all dependencies from requirements.txt
- Install system-specific dependencies if needed

### Quick Start

For a quick overview of the dataset and some basic visualizations:

```
python scripts/quick_start.py
```

This script will:
- Display basic information about the dataset
- Show summary statistics and value counts for key columns
- Generate 5 basic visualizations in the `output/quick_start_viz` directory

## Analysis Components

The project consists of three main components:

1. **Basic Exploratory Data Analysis** (`scripts/online_foods_eda.py`)
2. **Interactive Dashboard** (`scripts/online_foods_dashboard.py`)
3. **Machine Learning Analysis** (`scripts/online_foods_ml.py`)

### 1. Basic Exploratory Data Analysis

The analysis includes:

- **Basic Data Exploration**
   - Dataset overview and summary statistics
   - Missing value analysis
   - Data type inspection

- **Univariate Analysis**
   - Distribution of demographic variables (Age, Gender, Marital Status)
   - Distribution of socioeconomic variables (Occupation, Income, Education)
   - Distribution of online food service usage and feedback

- **Bivariate Analysis**
   - Relationship between demographic factors and online food service usage
   - Relationship between socioeconomic factors and online food service usage
   - Relationship between family size and online food service usage

- **Geographic Analysis**
   - Map visualization of online food service usage
   - Heatmap of online food service usage

- **Advanced Visualizations**
   - Interactive Plotly visualizations
   - 3D scatter plots
   - Hierarchical views using sunburst charts

- **Statistical Tests**
   - Chi-square tests for categorical variables
   - T-tests for numerical variables

### 2. Interactive Dashboard

The dashboard provides an interactive interface to explore the dataset with:

- **Filtering capabilities** by age, gender, marital status, and occupation
- **Multiple visualization tabs**:
  - Demographics
  - Socioeconomic Factors
  - Online Food Usage
  - Geographic Analysis
  - Advanced Analysis
- **Dynamic insights** that update based on the selected filters
- **Interactive maps** showing the geographic distribution of online food service usage

### 3. Machine Learning Analysis

The machine learning component builds predictive models for online food service usage:

- **Multiple models** are trained and evaluated:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- **Model evaluation** with metrics:
  - Accuracy, Precision, Recall, F1 Score, AUC
  - Confusion matrices
  - ROC curves
- **Feature importance analysis** to understand key factors
- **Hyperparameter tuning** for the best-performing model
- **Model comparison** to identify the most effective approach

## Setup and Running the Analysis

### Quick Setup

1. Run the installation script to install all dependencies:
   ```
   python scripts/install_dependencies.py
   ```

2. Run the setup script to check requirements and prepare the environment:
   ```
   python scripts/setup.py
   ```
   
   This script will:
   - Check if you have the required Python version (3.8+)
   - Verify that the CSV file exists
   - Create necessary directories for outputs

3. Run the complete analysis pipeline:
   ```
   python scripts/run_analysis.py
   ```

### Manual Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create the necessary directories:
   ```
   mkdir -p output/visualizations output/models output/dashboards output/quick_start_viz
   ```

3. Run the components separately:
   ```
   python scripts/online_foods_eda.py       # Basic EDA
   python scripts/online_foods_ml.py        # Machine Learning Analysis
   python scripts/online_foods_dashboard.py # Interactive Dashboard
   ```

### Output Files

After running the analysis, you'll find:
- Static visualizations (PNG files) in the `output/visualizations` directory
- Interactive visualizations (HTML files) in the `output/dashboards` directory
- Trained models (PKL files) in the `output/models` directory
- Quick start visualizations in the `output/quick_start_viz` directory

## Key Findings

The analysis reveals patterns in online food service usage across different demographic and socioeconomic groups, providing insights into:

- Which age groups are more likely to use online food services
- Gender differences in online food service usage
- Impact of income and education on online food service adoption
- Geographic patterns in online food service usage
- Relationship between family size and online food service usage
- Key predictive factors for online food service adoption

## Requirements

- Python 3.8 or higher
- See `requirements.txt` for the complete list of dependencies 
