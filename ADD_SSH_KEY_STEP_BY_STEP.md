# 🔑 ADD SSH KEY TO RUNPOD - Step by Step

## Your SSH Public Key (Copy This Entire Line!)

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPn/II7Hndfgq1tkLKv0qMlZCBTdG9Nd4EovXG5hVxJE omertahtoko@gmail.com
```

---

## 📋 Step-by-Step Instructions

### Step 1: Copy the SSH Key Above ☝️

Select the **entire line** starting with `ssh-ed25519` and copy it (Cmd+C)

### Step 2: Go to RunPod Settings

Open this URL in your browser:
**https://www.runpod.io/console/user/settings**

### Step 3: Navigate to SSH Keys

On the left sidebar, click: **"SSH Public Keys"**

### Step 4: Add New Key

Click the button: **"+ Add SSH Key"** or **"Add Public Key"**

### Step 5: Paste Your Key

- **Public Key field:** Paste the entire line you copied
- **Name/Label field:** Type `Mac SSH Key` (or any name you want)

### Step 6: Save

Click **"Add Key"** or **"Save"**

You should see your key appear in the list! ✅

---

## 🔄 Step 7: RESTART Your Pod (CRITICAL!)

**⚠️ The SSH key will NOT work until you restart the pod!**

1. Go to: **https://www.runpod.io/console/pods**
2. Find your pod with ID: **`pvj233wwhiu6j3-64411542`**
3. Click the **"Stop"** button (or three dots menu → Stop)
4. Wait for the pod to fully stop (status will change)
5. Click the **"Start"** button
6. Wait for the pod to fully start (green/running status)

**Note:** After restart:
- The SSH connection details might change
- Check the pod page for the current SSH command
- vLLM will need to be restarted

---

## ✅ Step 8: Test SSH Connection

Back in your Mac terminal, try again:

```bash
ssh pvj233wwhiu6j3-64411542@ssh.runpod.io -i ~/.ssh/id_ed25519
```

**Expected result:** You should be logged into RunPod!

```
Welcome to Ubuntu...
root@xxx:~#
```

If you see the RunPod shell prompt, **SUCCESS!** 🎉

---

## 🐛 Troubleshooting

### Still getting "Permission denied"?

**Check these:**

1. **Did you copy the ENTIRE key?**
   - Must include `ssh-ed25519` at the start
   - Must include the email at the end
   - No extra spaces or line breaks

2. **Did you RESTART the pod?**
   - Key only takes effect after pod restart
   - Stop → Wait → Start

3. **Is the pod running?**
   - Check pod status in RunPod dashboard
   - Should be green/running

4. **Try the direct TCP method:**
   - Check your pod page for the direct SSH command
   - Might look like: `ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519`

### Alternative: Use RunPod Web Terminal

If SSH still doesn't work:
1. Go to your pod page
2. Click **"Connect"** → **"Start Web Terminal"**
3. Use the browser-based terminal to start vLLM
4. Create SSH tunnel separately

---

## 📝 Quick Checklist

- [ ] Copied the entire SSH public key
- [ ] Went to RunPod user settings
- [ ] Added the SSH key
- [ ] Named it "Mac SSH Key"
- [ ] Saved the key
- [ ] Went to pods page
- [ ] Stopped the pod
- [ ] Waited for it to stop
- [ ] Started the pod
- [ ] Waited for it to start (green status)
- [ ] Tested SSH connection from Mac terminal
- [ ] Successfully logged in to RunPod

---

## 🎯 What's Next?

Once SSH works, you'll:
1. Start vLLM on RunPod
2. Create SSH tunnel from Mac
3. Start backend and frontend
4. Test the chatbot!

See `START_HERE.md` for the complete deployment flow.

---

**Need help? Check if the key is in the right format and pod is fully restarted!**
