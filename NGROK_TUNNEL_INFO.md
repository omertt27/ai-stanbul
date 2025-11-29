# ngrok Tunnel Information

## 🌐 Active Tunnel URL

```
https://boarishly-umbonic-archer.ngrok-free.dev
```

## ℹ️ Configuration Details

- **Auth Token**: Configured (found in `~/Library/Application Support/ngrok/ngrok.yml`)
- **ngrok Location**: `/opt/homebrew/bin/ngrok`
- **Target Port**: Unknown (need to verify which port the tunnel is forwarding)
- **Frontend Server**: Running on port **3000**
- **Tunnel Status**: Active (endpoint already in use)

## 🔍 Current Status

The ngrok tunnel is **already running** with the URL above. When trying to start a new tunnel, we got:

```
ERROR: The endpoint 'https://boarishly-umbonic-archer.ngrok-free.dev' is already online.
```

This means there's an existing ngrok process running somewhere.

## ✅ To Use This Tunnel

1. **Find the existing ngrok process**:
   ```bash
   ps aux | grep ngrok
   ```

2. **Check what port it's forwarding**:
   ```bash
   curl http://localhost:4040/api/tunnels
   ```
   This shows the ngrok web interface with tunnel details.

3. **If you need to restart ngrok**:
   ```bash
   # Kill existing ngrok
   pkill ngrok
   
   # Wait a moment
   sleep 2
   
   # Start new tunnel to your frontend (port 3000)
   ngrok http 3000
   ```

4. **For background operation**:
   ```bash
   nohup ngrok http 3000 > /tmp/ngrok.log 2>&1 &
   ```

## 🌐 Access ngrok Dashboard

Once ngrok is running, you can view the dashboard at:
```
http://localhost:4040
```

This shows:
- Current tunnel URL
- Request/response logs
- Traffic statistics
- Connection status

## 📝 Notes

- **Free tier**: URL changes every time ngrok restarts
- **Paid tier** ($8/mo): Get a permanent custom domain
- The current URL (`boarishly-umbonic-archer.ngrok-free.dev`) will work until you restart ngrok

## 🔧 Troubleshooting

### If tunnel returns 404:
1. Verify your frontend is running: `lsof -i :3000`
2. Check ngrok is forwarding to correct port: `curl http://localhost:4040/api/tunnels`
3. Restart ngrok if needed

### If you can't access localhost:4040:
- ngrok process might not be running
- Start ngrok: `ngrok http 3000`

### To get a new URL:
```bash
pkill ngrok
ngrok http 3000
```

## 🎯 Quick Commands

```bash
# Check if ngrok is running
ps aux | grep ngrok

# View tunnel URL
curl -s http://localhost:4040/api/tunnels | python3 -m json.tool

# Start ngrok for frontend
ngrok http 3000

# Start ngrok in background
nohup ngrok http 3000 > /tmp/ngrok.log 2>&1 &

# View ngrok logs
cat /tmp/ngrok.log

# Stop all ngrok processes
pkill ngrok
```

## 🌐 Your Active URL

**Use this URL to access your frontend:**
```
https://boarishly-umbonic-archer.ngrok-free.dev
```

Share this URL with others to let them access your local development server!
