param([string]$Path = ".azure_config.json")

if (-not (Test-Path $Path)) { Write-Error "Config not found: $Path"; exit 1 }
try { $cfg = Get-Content -Raw -Path $Path | ConvertFrom-Json } catch { Write-Error "Invalid JSON"; exit 1 }

$env:USE_LLM_OPEN = "1"
$env:LLM_BACKEND  = "azure"
$env:AZURE_OPENAI_ENDPOINT    = "$($cfg.endpoint)"
$env:AZURE_OPENAI_API_KEY     = "$($cfg.api_key)"
$env:AZURE_OPENAI_API_VERSION = "$($cfg.api_version)"
$env:AZURE_OPENAI_DEPLOYMENT  = "$($cfg.deployment)"
Write-Host "Azure env loaded from $Path"
