# take_screenshots.ps1 - opens the real tools and captures them with capture.ps1 (no editing of images).
# Run from the repository root in PowerShell:   .\setup\take_screenshots.ps1 [-Only q1|q2|stage1|files]
# Do not use the mouse/keyboard while it runs: whatever is on screen is captured.
param([string]$Only = "all", [int]$Pages = 40)

$Repo = (Resolve-Path "$PSScriptRoot\..").Path
$Shots = Join-Path $Repo "screenshots"
$Env = "C:\Users\IGIHOZO\anaconda3\envs\ml-project"
$Cap = Join-Path $PSScriptRoot "capture.ps1"
$Port = 8899; $Token = "group"
New-Item -ItemType Directory -Force $Shots | Out-Null
$wsh = New-Object -ComObject WScript.Shell
$edge = "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe"
if (-not (Test-Path $edge)) { $edge = "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe" }
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Mouse {
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")] public static extern void mouse_event(uint f, uint dx, uint dy, int data, UIntPtr extra);
    public static void Wheel(int x, int y, int notches) {     // scroll the window under (x, y); negative = down
        SetCursorPos(x, y);
        for (int i = 0; i < Math.Abs(notches); i++) { mouse_event(0x0800, 0, 0, notches < 0 ? -120 : 120, UIntPtr.Zero); System.Threading.Thread.Sleep(40); }
    }
}
"@

function Same-Image($a, $b) {  # true when two PNG files are identical (page did not move = end of notebook)
    if (-not (Test-Path $a) -or -not (Test-Path $b)) { return $false }
    return (Get-FileHash $a).Hash -eq (Get-FileHash $b).Hash
}

function Start-Jupyter {
    $running = $false
    try { Invoke-WebRequest -UseBasicParsing "http://localhost:$Port/api/status?token=$Token" -TimeoutSec 2 | Out-Null; $running = $true } catch {}
    if (-not $running) {
        $args = "-m jupyter notebook --no-browser --port $Port --IdentityProvider.token=$Token --ServerApp.root_dir=`"$Repo`""
        Start-Process -FilePath "$Env\python.exe" -ArgumentList $args -WindowStyle Minimized
        for ($i = 0; $i -lt 60; $i++) {
            Start-Sleep 2
            try { Invoke-WebRequest -UseBasicParsing "http://localhost:$Port/api/status?token=$Token" -TimeoutSec 2 | Out-Null; return } catch {}
        }
        throw "Jupyter did not start"
    }
}

function Shoot-Notebook($rel, $prefix) {
    # Opens one notebook in Jupyter Notebook (Edge app window) and captures it page by page.
    $name = [IO.Path]::GetFileNameWithoutExtension($rel)
    $url = "http://localhost:$Port/notebooks/$($rel -replace '\\','/')?token=$Token"
    Start-Process $edge "--new-window --start-maximized --app=$url"
    Start-Sleep 12                                              # let Jupyter load the notebook and its outputs
    $wsh.AppActivate($name) | Out-Null; Start-Sleep 1
    $wsh.SendKeys("{ESC}"); Start-Sleep 1                       # command mode (no cell is edited)
    $prev = $null
    for ($p = 1; $p -le $Pages; $p++) {
        $out = Join-Path $Shots ("{0}_{1}_p{2:00}.png" -f $prefix, $name, $p)
        & $Cap -Out $out -Title $name -DelayMs 700 | Out-Null
        if ($prev -and (Same-Image $prev $out)) { Remove-Item $out; break }   # reached the end
        $prev = $out
        [Mouse]::Wheel(1000, 700, -8); Start-Sleep 1             # scroll about one screen down (some overlap)
    }
    $wsh.AppActivate($name) | Out-Null; $wsh.SendKeys("%{F4}"); Start-Sleep 2   # close the window
    Write-Output "captured $name"
}

if ($Only -in "all", "stage1") {
    # Anaconda Prompt: activate ml-project and list environments
    Start-Process cmd.exe "/k title Anaconda Prompt (ml-project) && call C:\Users\IGIHOZO\anaconda3\Scripts\activate.bat ml-project && conda env list && python --version"
    Start-Sleep 15
    & $Cap -Out (Join-Path $Shots "stage1_anaconda_prompt.png") -Title "Anaconda Prompt" -DelayMs 800
    $wsh.AppActivate("Anaconda Prompt") | Out-Null; $wsh.SendKeys("exit{ENTER}")
    # Anaconda Navigator
    Start-Process "C:\Users\IGIHOZO\anaconda3\Scripts\anaconda-navigator.exe"
    Start-Sleep 90
    & $Cap -Out (Join-Path $Shots "stage1_anaconda_navigator.png") -Title "Anaconda Navigator" -DelayMs 1500
}

if ($Only -in "all", "stage1", "q1", "q2") { Start-Jupyter }
if ($Only -in "all", "stage1") { Shoot-Notebook "q1_maize_yield\notebooks\00_environment_check.ipynb" "q1" }
if ($Only -in "all", "q1") {
    foreach ($n in "01_acquire", "02_eda", "03_clean", "04_features") { Shoot-Notebook "q1_maize_yield\notebooks\$n.ipynb" "q1" }
}
if ($Only -in "all", "q2") {
    foreach ($n in "01_acquire", "02_eda", "03_clean", "04_features") { Shoot-Notebook "q2_own_scenario\notebooks\$n.ipynb" "q2" }
}
if ($Only -in "all", "files") {
    foreach ($pair in @(@("q1_maize_yield\data\processed", "q1_processed"), @("q2_own_scenario\data\processed", "q2_processed"),
                        @("q2_own_scenario\data\raw\gsod", "q2_raw_files"))) {
        Start-Process explorer.exe (Join-Path $Repo $pair[0]); Start-Sleep 4
        & $Cap -Out (Join-Path $Shots "files_$($pair[1]).png") -Title (Split-Path -Leaf $pair[0]) -DelayMs 800
        $wsh.AppActivate((Split-Path -Leaf $pair[0])) | Out-Null; $wsh.SendKeys("%{F4}"); Start-Sleep 1
    }
}
Write-Output "Screenshots saved in $Shots"
