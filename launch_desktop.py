import ctypes
from ctypes import wintypes
import sys
import os

class STARTUPINFO(ctypes.Structure):
    _fields_ = [
        ('cb', wintypes.DWORD),
        ('lpReserved', wintypes.LPWSTR),
        ('lpDesktop', wintypes.LPWSTR),
        ('lpTitle', wintypes.LPWSTR),
        ('dwX', wintypes.DWORD),
        ('dwY', wintypes.DWORD),
        ('dwXSize', wintypes.DWORD),
        ('dwYSize', wintypes.DWORD),
        ('dwXCountChars', wintypes.DWORD),
        ('dwYCountChars', wintypes.DWORD),
        ('dwFillAttribute', wintypes.DWORD),
        ('dwFlags', wintypes.DWORD),
        ('wShowWindow', wintypes.WORD),
        ('cbReserved2', wintypes.WORD),
        ('lpReserved2', ctypes.c_char_p),
        ('hStdInput', wintypes.HANDLE),
        ('hStdOutput', wintypes.HANDLE),
        ('hStdError', wintypes.HANDLE),
    ]

class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ('hProcess', wintypes.HANDLE),
        ('hThread', wintypes.HANDLE),
        ('dwProcessId', wintypes.DWORD),
        ('dwThreadId', wintypes.DWORD),
    ]

si = STARTUPINFO()
si.cb = ctypes.sizeof(STARTUPINFO)
si.lpDesktop = "WinSta0\\Default"

pi = PROCESS_INFORMATION()

py_exe = sys.executable
app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_windows.py")
cmd = f'"{py_exe}" "{app_path}"'
cwd = os.path.dirname(os.path.abspath(__file__))

# CREATE_NEW_CONSOLE = 0x00000010
res = ctypes.windll.kernel32.CreateProcessW(
    None, cmd, None, None, False, 0x00000010, None, cwd, ctypes.byref(si), ctypes.byref(pi)
)

if not res:
    err = ctypes.windll.kernel32.GetLastError()
    print(f"Error launching on WinSta0\\Default: {err}")
    sys.exit(1)
else:
    print(f"Successfully started PID {pi.dwProcessId} on desktop WinSta0\\Default")
