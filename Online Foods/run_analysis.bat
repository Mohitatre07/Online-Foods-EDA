@echo off
echo ===================================================
echo Online Foods Exploratory Data Analysis
echo ===================================================
echo.

echo Step 1: Installing dependencies...
python scripts\install_dependencies.py
if %ERRORLEVEL% NEQ 0 (
    echo Error installing dependencies!
    pause
    exit /b 1
)

echo.
echo Step 2: Setting up the environment...
python scripts\setup.py
if %ERRORLEVEL% NEQ 0 (
    echo Error setting up the environment!
    pause
    exit /b 1
)

echo.
echo Step 3: Running the analysis...
python scripts\run_analysis.py
if %ERRORLEVEL% NEQ 0 (
    echo Error running the analysis!
    pause
    exit /b 1
)

echo.
echo ===================================================
echo Analysis completed successfully!
echo ===================================================
echo.
echo Check the output directory for results.
echo.
pause 