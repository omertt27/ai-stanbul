#!/usr/bin/env python3
"""
Deploy fixed server to RunPod via SSH
Handles SSH commands programmatically
"""

import subprocess
import time
import os

POD_HOST = "e9e56rc2ryjtmm-64411022@ssh.runpod.io"
SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519")
LOCAL_FILE = "/Users/omer/Desktop/ai-stanbul/runpod_server_fixed.py"

def run_ssh_command(command, show_output=True):
    """Run SSH command and return output"""
    ssh_cmd = [
        "ssh",
        "-i", SSH_KEY,
        "-o", "StrictHostKeyChecking=no",
        "-T",  # No PTY
        POD_HOST,
        command
    ]
    
    result = subprocess.run(ssh_cmd, capture_output=True, text=True)
    
    if show_output:
        if result.stdout:
            print(result.stdout)
        if result.returncode != 0 and result.stderr and "PTY" not in result.stderr:
            print(f"Error: {result.stderr}")
    
    return result

def upload_file():
    """Upload fixed server file"""
    print("📤 Uploading fixed server...")
    
    # Use stdin redirection
    with open(LOCAL_FILE, 'r') as f:
        ssh_cmd = [
            "ssh",
            "-i", SSH_KEY,
            "-o", "StrictHostKeyChecking=no",
            "-T",
            POD_HOST,
            "cat > /workspace/server_fixed.py"
        ]
        result = subprocess.run(ssh_cmd, stdin=f, capture_output=True, text=True)
        
        if result.returncode == 0 or "PTY" in result.stderr:
            print("✅ File uploaded")
            return True
        else:
            print(f"❌ Upload failed: {result.stderr}")
            return False

def deploy():
    """Main deployment flow"""
    print("🚀 RunPod Server Fix Deployment")
    print("=" * 50)
    
    # Upload
    if not upload_file():
        return False
    
    # Backup and replace
    print("\n📋 Backing up current server...")
    run_ssh_command("cp /workspace/server.py /workspace/server.py.backup.$(date +%Y%m%d_%H%M%S)")
    
    print("🔄 Replacing server...")
    run_ssh_command("cp /workspace/server_fixed.py /workspace/server.py")
    
    # Restart
    print("\n🔄 Restarting server...")
    run_ssh_command("pkill -f 'python.*server.py' || true")
    time.sleep(2)
    
    run_ssh_command("cd /workspace && nohup python server.py > logs/server.log 2>&1 &")
    
    print("\n⏳ Waiting for server to start...")
    time.sleep(8)
    
    # Check process
    print("\n🔍 Checking server process...")
    result = run_ssh_command("ps aux | grep 'server.py' | grep -v grep || echo 'No process found'")
    
    print("\n✅ Deployment complete!")
    print("\nTest the server with:")
    print("  bash test_runpod_server.sh")
    
    return True

if __name__ == "__main__":
    deploy()
