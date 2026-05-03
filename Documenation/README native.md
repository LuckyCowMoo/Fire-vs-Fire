# Native Host (Windows) — Local AI Classifier (Dev Echo)

This folder provides a minimal native messaging host that echoes back neutral classification results so you can test the end‑to‑end pipeline.

The background script expects the host name `com.example.localclassifier` (configure in `background/background.js`).

## Files

- `echo_host.py`: Simple Python host that reads one message, returns a `classifyResult` with `uncertain` labels.
- `manifest.firefox.json`: Firefox manifest using `allowed_extensions` for the add‑on id `template@example.com`.
- `manifest.chrome.json`: Chrome manifest using `allowed_origins` (update with your Chrome extension id).
- `register-firefox-host.reg`: Windows registry template to register the Firefox host.
- `register-chrome-host.reg`: Windows registry template to register the Chrome host.

## Prerequisites

- Python 3 installed (adjust the path in the manifests):
  - Update the `path` field to your Python executable (e.g., `C:\\Users\\<you>\\AppData\\Local\\Programs\\Python\\Python311\\python.exe`).
- Ensure the `arguments` script path is valid. The current manifests assume the script is in `native\\echo_host.py` relative to where you place the manifest. If you move files, update the manifest accordingly.

## Install (Firefox)

Option A — Quick installer:

- Run the PowerShell installer, which auto‑detects Python, fixes manifest paths, and writes the registry key.

```powershell
# From the repo root
Set-Location "c:\Stuff\coding\Test Project\native"; powershell -ExecutionPolicy Bypass -File .\install-firefox.ps1
# Or with a custom Python path
Set-Location "c:\Stuff\coding\Test Project\native"; powershell -ExecutionPolicy Bypass -File .\install-firefox.ps1 -PythonPath "C:\\Path\\To\\Python\\python.exe"
```

Option B — Manual:

1. Edit `manifest.firefox.json`:
   - `path`: absolute path to your `python.exe`.
   - `arguments`: absolute path to `echo_host.py`.
   - `allowed_extensions`: should match the add‑on id `template@example.com`.
2. Double‑click `register-firefox-host.reg` (or import via `regedit`). Update the path inside if your repo location differs.
3. Restart Firefox (or disable/enable the extension).

Registry key that will be created:

```
HKEY_CURRENT_USER\Software\Mozilla\NativeMessagingHosts\com.example.localclassifier
```

## Install (Chrome / Edge)

1. Load/unpack the extension and note the extension ID from `chrome://extensions`.
2. Edit `manifest.chrome.json`:
   - `path`: absolute path to your `python.exe`.
   - `arguments`: absolute or relative path to `echo_host.py` as you prefer.
   - `allowed_origins`: replace `YOUR_EXTENSION_ID` with the actual id, e.g., `chrome-extension://abc123def456ghijklmno/`.
3. Double‑click `register-chrome-host.reg` (or import via `regedit`).
4. Restart Chrome.

Registry key that will be created:

```
HKEY_CURRENT_USER\Software\Google\Chrome\NativeMessagingHosts\com.example.localclassifier
```

## Test the pipeline

- Toggle a dev fallback first if you want to test without installing the host: set `ENABLE_FAKE_NATIVE = true` in `background/background.js`.
- Otherwise:
  1. Ensure the registry points to the correct manifest file.
  2. Ensure the manifest `path` points to a working `python.exe`.
  3. Optional: Validate the host locally without the browser:

     ```powershell
     # Run the local test harness
     Set-Location "c:\Stuff\coding\Test Project\native"; python .\test_host.py
     ```

     You should see a `classifyResult` response printed.

  4. Open a page; the content script will extract items and send them. Check the browser console:
     - You should see `[Extension] Classification results` with a `classifyResult` payload.
  5. If you see `Native messaging API unavailable`, confirm:
     - `nativeMessaging` is present in `manifest.json` permissions.
     - You’re testing on desktop Firefox/Chrome (not Android).
     - Background script is MV2 (Firefox) or MV3 service worker (Chrome) with the right API (this project uses MV2).
     - Host manifest is registered correctly and readable by the browser.

## Troubleshooting

- If responses don’t arrive:
  - Run `echo_host.py` in a terminal to spot syntax issues.
  - Check Windows Event Viewer for application errors if the host crashes instantly.
  - Use absolute paths in `arguments` to avoid working‑directory issues.
  - On Firefox, unsigned temporary add‑ons support native messaging, but you still must register the host.
