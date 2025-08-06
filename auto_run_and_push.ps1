# Auto Run and Push Script for Insurance Bot

Write-Host "🚀 Starting Auto Run and Push Script..." -ForegroundColor Green

# 1. Activate virtual environment
Write-Host "1️⃣ Activating virtual environment..." -ForegroundColor Cyan
.\venv\Scripts\Activate.ps1

# 2. Install/Update dependencies
Write-Host "2️⃣ Installing/Updating dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt

# 3. Run the application tests
Write-Host "3️⃣ Running the application..." -ForegroundColor Cyan
python hackrx_final_demo.py

# 4. Git operations
Write-Host "4️⃣ Performing Git operations..." -ForegroundColor Cyan

# Initialize Git if needed
if (-not (Test-Path .git)) {
    Write-Host "Initializing Git repository..." -ForegroundColor Yellow
    git init
}

# Add all files
Write-Host "Adding files to Git..." -ForegroundColor Yellow
git add .

# Commit changes
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$commit_message = "Auto update: $timestamp"
Write-Host "Committing changes..." -ForegroundColor Yellow
git commit -m $commit_message

# Push to remote
Write-Host "Pushing to remote repository..." -ForegroundColor Yellow
git push origin main

Write-Host "✅ Script completed successfully!" -ForegroundColor Green
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
