# capture.ps1 - takes a REAL screenshot of a window (or the whole screen) and saves it as PNG.
# Usage examples (PowerShell):
#   .\setup\capture.ps1 -Out screenshots\fig01_navigator.png -Title "Anaconda Navigator"
#   .\setup\capture.ps1 -Out screenshots\fig02_prompt.png -Title "Anaconda Prompt"
#   .\setup\capture.ps1 -Out screenshots\full.png            # whole primary screen
# -Title matches part of the window title; the window is brought to the front before capturing.
param(
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$Title = "",
    [int]$DelayMs = 800
)
Add-Type -AssemblyName System.Drawing, System.Windows.Forms
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Win {
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int cmd);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern int GetWindowText(IntPtr h, System.Text.StringBuilder s, int n);
    public delegate bool EnumProc(IntPtr h, IntPtr p);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc cb, IntPtr p);
    public struct RECT { public int Left, Top, Right, Bottom; }
    public static IntPtr Find(string part) {          // first visible top-level window whose title contains 'part'
        IntPtr found = IntPtr.Zero;
        EnumWindows((h, p) => {
            if (!IsWindowVisible(h)) return true;
            var sb = new System.Text.StringBuilder(512); GetWindowText(h, sb, 512);
            string t = sb.ToString();
            if (t.IndexOf("Visual Studio Code", StringComparison.OrdinalIgnoreCase) >= 0) return true;
            if (t.IndexOf(part, StringComparison.OrdinalIgnoreCase) >= 0) { found = h; return false; }
            return true; }, IntPtr.Zero);
        return found;
    }
}
"@
[Win]::SetProcessDPIAware() | Out-Null
if (-not [IO.Path]::IsPathRooted($Out)) { $Out = Join-Path (Get-Location).Path $Out }
$Out = [IO.Path]::GetFullPath($Out)
$dir = Split-Path -Parent $Out
if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force $dir | Out-Null }

if ($Title) {
    $h = [Win]::Find($Title)
    if ($h -eq [IntPtr]::Zero) { throw "No window whose title contains '$Title'." }
    [Win]::ShowWindow($h, 3) | Out-Null   # 3 = maximise
    [Win]::SetForegroundWindow($h) | Out-Null
    Start-Sleep -Milliseconds $DelayMs
    $r = New-Object Win+RECT
    [Win]::GetWindowRect($h, [ref]$r) | Out-Null
    $x = [Math]::Max($r.Left, 0); $y = [Math]::Max($r.Top, 0)
    $w = $r.Right - $x; $h = $r.Bottom - $y
} else {
    Start-Sleep -Milliseconds $DelayMs
    $b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
    $x = $b.X; $y = $b.Y; $w = $b.Width; $h = $b.Height
}
$bmp = New-Object System.Drawing.Bitmap $w, $h
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($x, $y, 0, 0, $bmp.Size)
$bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()
Write-Output "saved $Out ($w x $h)"
