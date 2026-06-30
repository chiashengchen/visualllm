// Tiny native launcher for VisualLLm. Double-clicking the compiled .exe opens a
// console and hands off to scripts\launch.ps1 (which brings the whole stack up).
// Compiled with the csc.exe that ships with Windows -- no external build tooling.
// Build: scripts\build-exe.ps1
using System;
using System.Diagnostics;
using System.IO;

class Launcher
{
    static int Main()
    {
        // The .exe lives at the repo root; the orchestrator is scripts\launch.ps1.
        string exeDir = AppDomain.CurrentDomain.BaseDirectory;
        string script = Path.Combine(exeDir, "scripts", "launch.ps1");
        if (!File.Exists(script))
        {
            // Fallback: exe sitting inside scripts\ next to launch.ps1.
            script = Path.Combine(exeDir, "launch.ps1");
        }
        if (!File.Exists(script))
        {
            Console.Error.WriteLine("Cannot find scripts\\launch.ps1 next to this .exe.");
            Console.Error.WriteLine("Keep the .exe in the VisualLLm project root.");
            Console.Error.WriteLine("Press any key to exit...");
            Console.ReadKey();
            return 1;
        }

        var psi = new ProcessStartInfo
        {
            FileName = "powershell.exe",
            Arguments = "-ExecutionPolicy Bypass -NoProfile -File \"" + script + "\"",
            UseShellExecute = false
        };
        try
        {
            var p = Process.Start(psi);
            p.WaitForExit();
            return p.ExitCode;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine("Failed to start PowerShell: " + ex.Message);
            Console.ReadKey();
            return 1;
        }
    }
}
