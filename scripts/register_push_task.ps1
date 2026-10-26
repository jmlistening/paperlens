#Requires -Version 5.1
<#
.SYNOPSIS
    Register the daily PaperLens frontier push task.

.DESCRIPTION
    Creates or updates the scheduled task "PaperLens Frontier Push", which runs
    scripts\run_push.bat daily at the configured local time. The task runs only
    in the signed-in user's interactive session. No secret is stored on the
    task: Python reads the .env file at runtime.

    Settings: StartWhenAvailable, MultipleInstances=IgnoreNew,
    RestartCount=2, RestartInterval=10 minutes, ExecutionTimeLimit=2 hours.

.PARAMETER TaskName
    Scheduled task name. Default: "PaperLens Frontier Push".

.PARAMETER ProjectDir
    PaperLens project directory. Default: the checkout containing this script.

.PARAMETER Time
    Daily local time as HH:mm. Defaults to push_config.json "time_local".

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\register_push_task.ps1 -WhatIf

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\register_push_task.ps1
#>
[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Medium')]
param(
    [string]$TaskName = 'PaperLens Frontier Push',
    [string]$ProjectDir = '',
    [string]$Time = ''
)

$ErrorActionPreference = 'Stop'

if (-not $ProjectDir) {
    $ProjectDir = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
}

if (-not $Time) {
    $configPath = Join-Path $ProjectDir 'push_config.json'
    if (Test-Path -LiteralPath $configPath) {
        $config = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($config.time_local) {
            $Time = [string]$config.time_local
        }
    }
}
if (-not $Time) {
    $Time = '08:30'
}

$atTime = [datetime]::ParseExact($Time, 'HH:mm', [System.Globalization.CultureInfo]::InvariantCulture)

$batchPath = Join-Path $ProjectDir 'scripts/run_push.bat'
if (-not (Test-Path -LiteralPath $batchPath)) {
    throw "Missing batch entrypoint: $batchPath"
}

$action = New-ScheduledTaskAction `
    -Execute 'cmd.exe' `
    -Argument ('/c "{0}"' -f $batchPath) `
    -WorkingDirectory $ProjectDir

$trigger = New-ScheduledTaskTrigger -Daily -At $atTime

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 10) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal `
    -UserId $currentUser `
    -LogonType Interactive `
    -RunLevel Limited

if ($PSCmdlet.ShouldProcess($TaskName, "Register daily scheduled task at $Time")) {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Force | Out-Null
    Write-Host "Registered '$TaskName' daily at $Time (interactive logon)."
    Write-Host 'Use unregister_push_task.ps1 to remove it.'
}
