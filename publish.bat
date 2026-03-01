@echo off
echo ========================================
echo EssentiaX v1.1.1 - Publishing Script
echo ========================================
echo.

echo Step 1: Checking build files...
if not exist "dist\essentiax-1.1.1-py3-none-any.whl" (
    echo ERROR: Build files not found!
    echo Please run: python setup.py sdist bdist_wheel
    pause
    exit /b 1
)
echo ✅ Build files found!
echo.

echo Step 2: Ready to upload to PyPI
echo.
echo Choose an option:
echo 1. Upload to TestPyPI (recommended for testing)
echo 2. Upload to PyPI (production)
echo 3. Cancel
echo.
set /p choice="Enter your choice (1, 2, or 3): "

if "%choice%"=="1" (
    echo.
    echo Uploading to TestPyPI...
    twine upload --repository testpypi dist/*
    echo.
    echo ✅ Uploaded to TestPyPI!
    echo.
    echo Test installation with:
    echo pip install --index-url https://test.pypi.org/simple/ Essentiax
) else if "%choice%"=="2" (
    echo.
    echo ⚠️  WARNING: This will publish to production PyPI!
    echo This action cannot be undone.
    echo.
    set /p confirm="Are you sure? (yes/no): "
    if /i "%confirm%"=="yes" (
        echo.
        echo Uploading to PyPI...
        twine upload dist/*
        echo.
        echo ✅ Published to PyPI!
        echo.
        echo Users can now install with:
        echo pip install --upgrade Essentiax
    ) else (
        echo.
        echo ❌ Upload cancelled.
    )
) else (
    echo.
    echo ❌ Upload cancelled.
)

echo.
echo ========================================
echo Done!
echo ========================================
pause
