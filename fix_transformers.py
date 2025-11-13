import subprocess, sys, shutil, importlib, os

site_dir = "/usr/local/lib/python3.10/site-packages/transformers"
if os.path.exists(site_dir):
    print("🧹 Removing preinstalled transformers...")
    shutil.rmtree(site_dir, ignore_errors=True)

print("📦 Installing correct version of transformers...")
subprocess.run([sys.executable, "-m", "pip", "install", "-U", "transformers==4.46.1", "huggingface-hub>=0.24.0"], check=True)
importlib.invalidate_caches()
print("✅ Transformers fixed.")
