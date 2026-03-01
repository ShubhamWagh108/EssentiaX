@echo off
echo ========================================
echo EssentiaX v1.1.1 - Git Publishing
echo ========================================
echo.

echo Step 1: Adding all changes...
git add .
if %errorlevel% neq 0 (
    echo ERROR: Failed to add files
    pause
    exit /b 1
)
echo ✅ Files added
echo.

echo Step 2: Committing changes...
git commit -m "Release v1.1.1: Fixed Colab visualization display + Advanced 3D visualizations"
if %errorlevel% neq 0 (
    echo ERROR: Failed to commit
    pause
    exit /b 1
)
echo ✅ Changes committed
echo.

echo Step 3: Pushing to GitHub...
git push origin main
if %errorlevel% neq 0 (
    echo ERROR: Failed to push
    pause
    exit /b 1
)
echo ✅ Pushed to GitHub
echo.

echo Step 4: Creating tag v1.1.1...
git tag v1.1.1
if %errorlevel% neq 0 (
    echo WARNING: Tag might already exist
    echo Deleting old tag...
    git tag -d v1.1.1
    git push origin :refs/tags/v1.1.1
    git tag v1.1.1
)
echo ✅ Tag created
echo.

echo Step 5: Pushing tag to GitHub...
git push origin v1.1.1
if %errorlevel% neq 0 (
    echo ERROR: Failed to push tag
    pause
    exit /b 1
)
echo ✅ Tag pushed to GitHub
echo.

echo ========================================
echo ✅ SUCCESS!
echo ========================================
echo.
echo Next steps:
echo 1. Go to your GitHub repository
echo 2. Click on "Releases"
echo 3. Click "Create a new release"
echo 4. Select tag: v1.1.1
echo 5. Add release notes from V1.1.1_RELEASE_NOTES.md
echo 6. Click "Publish release"
echo.
echo GitHub Actions will automatically publish to PyPI!
echo.
pause
