<#
Register (or update) the "ZotWatch Profile Refresh" scheduled task for the current user.

Runs tools\refresh_profile.ps1 every Thursday 13:00, two hours before the 15:00 digest.
StartWhenAvailable: a run missed because the PC was off starts as soon as it is back.
Runs only while this user is logged on, so no password is stored and no admin is needed.
Re-running this script replaces the task with the same definition.

Remove it with:  Unregister-ScheduledTask -TaskName 'ZotWatch Profile Refresh' -Confirm:$false
#>
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$script = Join-Path $repo 'tools\refresh_profile.ps1'
if (-not (Test-Path $script)) { throw "refresh script not found: $script" }

$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument ('-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}"' -f $script) `
    -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Thursday -At '13:00'
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName 'ZotWatch Profile Refresh' -Action $action -Trigger $trigger `
    -Settings $settings -Force `
    -Description 'Rebuild the ZotWatch library profile from local Zotero and upload it before the Thursday 15:00 digest. Logs: logs\refresh-profile-*.log' |
    Out-Null

$task = Get-ScheduledTask -TaskName 'ZotWatch Profile Refresh'
$info = $task | Get-ScheduledTaskInfo
'registered: {0} | state: {1} | next run: {2:yyyy-MM-dd HH:mm} | user: {3} | logon: {4}' -f `
    $task.TaskName, $task.State, $info.NextRunTime, $task.Principal.UserId, $task.Principal.LogonType
