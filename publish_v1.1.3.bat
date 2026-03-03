@echo off
REM EssentiaX v1.1.3 Publishing Script for Windows
REM ===============================================

echo.
echo ========================================
echo  EssentiaX v1.1.3 Publishing Script
echo ========================================
echo.

REM Step 1: Clean previous builds
echo [1/6] Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist Essentiax.egg-info rmdir /s /q Essentiax.egg-info
echo     Done!
echo.

REM Step 2: Upgrade build tools
echo [2/6] Upgrading build tools...
pip install --upgrade setuptools wheel twine
echo     Done!
echo.

REM Step 3: Build package
echo [3/6] Building package...
python setup.py sdist bdist_wheel
echo     Done!
echo.

REM Step 4: Check build
echo [4/6] Verifying build...
if exist dist\Essentiax-1.1.3-py3-none-any.whl (
    echo     ✓ Wheel file created
) else (
    echo     ✗ Wheel file NOT found!
    pause
    exit /b 1
)

if exist dist\Essentiax-1.1.3.tar.gz (
    echo     ✓ Source distribution created
) else (
    echo     ✗ Source distribution NOT found!
    pause
    exit /b 1
)
echo.

REM Step 5: Upload to PyPI
echo [5/6] Uploading to PyPI...
echo.
echo     You will be prompted for your PyPI credentials:
echo     Username: __token__
echo     Password: [your-pypi-token]
echo.
twine upload dist/*
echo     Done!
echo.

REM Step 6: Verify
echo [6/6] Verification steps...
echo.
echo     Please verify:
echo     1. Visit: https://pypi.org/project/essentiax/
echo     2. Check version shows as 1.1.3
echo     3. Test installation: pip install --upgrade essentiax
echo     4. Test in Google Colab
echo.

echo ========================================
echo  Publishing Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Test in Colab: https://colab.research.google.com/
echo 2. Create GitHub release
echo 3. Update documentation
echo.

pause
