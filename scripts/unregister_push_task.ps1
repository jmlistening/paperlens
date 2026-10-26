#Requires -Version 5.1
<#
.SYNOPSIS
    Remove the daily PaperLens frontier push task.

.DESCRIPTION
    Safely unregisters "PaperLens Frontier Push" when it exists and does
    nothing when it is already absent. Supports -WhatIf.

.PARAMETER TaskName
    Scheduled task name. Default: "PaperLens Frontier Push".

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\unregister_push_task.ps1 -WhatIf

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\unregister_push_task.ps1
#>
[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Medium')]
param(
    [string]$TaskName = 'PaperLens Frontier Push'
)

$ErrorActionPreference = 'Stop'

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -eq $task) {
    Write-Host "Task '$TaskName' is not registered; nothing to remove."
    exit 0
}

if ($PSCmdlet.ShouldProcess($TaskName, 'Unregister scheduled task')) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed scheduled task '$TaskName'."
}
