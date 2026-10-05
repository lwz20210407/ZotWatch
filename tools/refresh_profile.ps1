<#
Weekly profile refresh for ZotWatch.

Rebuilds the profile from the local Zotero library (zotero.sqlite is copied before it is
read, so Zotero may stay open), verifies it, and publishes it as the `profile-latest`
release asset that the Thursday 15:00 run downloads. Only new or changed items are
re-embedded, so a normal week costs seconds of embedding and one ~30 MB upload.

Run by the "ZotWatch Profile Refresh" scheduled task every Thursday 13:00; safe to run
by hand at any time:
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\refresh_profile.ps1

The previous bundle is kept as data\profile-bundle.prev.tar.gz. To roll back:
    Copy-Item data\profile-bundle.prev.tar.gz $env:TEMP\profile-bundle.tar.gz
    gh release upload profile-latest $env:TEMP\profile-bundle.tar.gz --clobber --repo lwz20210407/ZotWatch
Logs: logs\refresh-profile-YYYYMMDD-HHMMSS.log (the newest 12 are kept).
Exit code 0 = refreshed and uploaded; 1 = failed, the previous published bundle stays.
#>
$ErrorActionPreference = 'Continue'   # native tools write progress to stderr; exit codes decide
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = '1'

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$logDir = Join-Path $repo 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ('refresh-profile-{0:yyyyMMdd-HHmmss}.log' -f (Get-Date))
$python = Join-Path $repo '.venv\Scripts\python.exe'
$bundle = Join-Path $repo 'data\profile-bundle.tar.gz'

function Write-Log([string]$message) {
    ('{0:yyyy-MM-dd HH:mm:ss} {1}' -f (Get-Date), $message) | Out-File -FilePath $log -Append -Encoding utf8
}

function Invoke-Step([string]$name, [scriptblock]$command) {
    Write-Log "== $name"
    & $command 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath $log -Append -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw "$name failed (exit $LASTEXITCODE)" }
}

$code = 0
$started = Get-Date
try {
    if (-not (Test-Path $python)) { throw "project venv not found: $python" }
    if (Test-Path $bundle) {
        Copy-Item $bundle (Join-Path $repo 'data\profile-bundle.prev.tar.gz') -Force
        Write-Log 'kept the previous bundle as data\profile-bundle.prev.tar.gz'
    }
    Invoke-Step 'build profile from local Zotero' { & $python -m src.cli profile --local --bundle }
    Invoke-Step 'verify profile' { & $python -m src.cli verify-profile }

    $uploaded = $false
    foreach ($attempt in 1..3) {
        try {
            Invoke-Step "upload bundle (attempt $attempt)" {
                & gh release upload profile-latest $bundle --clobber --repo lwz20210407/ZotWatch
            }
            $uploaded = $true
            break
        } catch {
            Write-Log "$_"
            Start-Sleep -Seconds (30 * $attempt)
        }
    }
    if (-not $uploaded) { throw 'upload failed after 3 attempts' }
    Write-Log ('OK: profile refreshed and uploaded in {0:N0} s' -f ((Get-Date) - $started).TotalSeconds)
} catch {
    Write-Log "FAILED: $_"
    $code = 1
} finally {
    Get-ChildItem $logDir -Filter 'refresh-profile-*.log' | Sort-Object Name -Descending |
        Select-Object -Skip 12 | Remove-Item -Force
}
exit $code
