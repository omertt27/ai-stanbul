# 🔐 Fix RDS Security Group - Step by Step

## ✅ Your RDS Configuration (Confirmed)

- **Endpoint**: database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com
- **Port**: 5432
- **VPC**: vpc-0cbb00ea173822ceb
- **Security Group**: sg-097205416c4b1f1e9 (default)
- **Publicly Accessible**: ✅ **YES** (Good!)
- **Your IP Address**: 159.20.69.7

---

## 🎯 What You Need to Do

The RDS instance is publicly accessible, but the **security group is blocking your IP**. 

You need to add **one inbound rule** to allow PostgreSQL connections from your IP.

---

## 📋 Step-by-Step Instructions

### **Option 1: Via Security Group Link (Fastest)**

1. **Click this direct link to your security group**:
   
   https://console.aws.amazon.com/ec2/v2/home?region=eu-central-1#SecurityGroup:securityGroupId=sg-097205416c4b1f1e9

2. **Click "Edit inbound rules"** button (bottom right area)

3. **Click "Add rule"** button

4. **Fill in the new rule**:
   ```
   Type: PostgreSQL
   Protocol: TCP (auto-filled)
   Port range: 5432 (auto-filled)
   Source: Custom → 159.20.69.7/32
   Description: Local development access
   ```

5. **Click "Save rules"**

---

### **Option 2: Via RDS Console (Alternative)**

1. **Go to your RDS instance**:
   
   https://console.aws.amazon.com/rds/home?region=eu-central-1#database:id=database-1

2. **Under "Connectivity & security" tab**:
   - Find "VPC security groups"
   - Click on **"default (sg-097205416c4b1f1e9)"**

3. **Follow steps 2-5 from Option 1 above**

---

## 🖼️ Visual Guide

When editing inbound rules, your new rule should look like this:

```
┌─────────────────────────────────────────────────────────────┐
│  Type        │ Protocol │ Port Range │ Source              │
├──────────────┼──────────┼────────────┼─────────────────────┤
│ PostgreSQL   │ TCP      │ 5432       │ 159.20.69.7/32     │
│              │          │            │ [Custom]            │
└─────────────────────────────────────────────────────────────┘
Description: Local development access
```

---

## ⏱️ After Saving

- Rules take effect **immediately** (but allow 1-2 minutes for propagation)
- You'll see the rule appear in the "Inbound rules" list

---

## 🧪 Test Connection (After 2 Minutes)

Run this command to test:

```bash
cd /Users/omer/Desktop/ai-stanbul
python3 test_aws_connection.py
```

### Expected Success Output:

```
✅ Connection successful!

📊 Database Info:
PostgreSQL Version: PostgreSQL 16.x on x86_64-pc-linux-gnu...
Tables in database: 0

✅ AWS RDS CONNECTION TEST PASSED!

🚀 Next step: Run the migration script
   python migrate_render_to_aws.py
```

---

## ❌ If Connection Still Fails

### Check Your IP Hasn't Changed:

```bash
curl ifconfig.me
```

If it's different from `159.20.69.7`, update the security group rule with the new IP.

### Common Issues:

1. **Wrong IP format**: Make sure you used `159.20.69.7/32` (with `/32` at the end)
2. **Wrong port**: Must be `5432`
3. **Wrong protocol**: Must be `TCP`
4. **Didn't save**: Click "Save rules" button!

---

## 🔒 Security Best Practices

### For Development (Now):
- Add your specific IP: `159.20.69.7/32`
- Update the rule if your IP changes

### For Production (Later):
- Remove your personal IP after migration
- Only allow Cloud Run IP ranges (or use VPC peering)
- Consider using a bastion host or VPN

---

## 🚀 After Security Group is Fixed

1. **Test connection**:
   ```bash
   python3 test_aws_connection.py
   ```

2. **Run migration**:
   ```bash
   python3 migrate_render_to_aws.py
   ```

3. **Verify data**:
   ```bash
   python3 verify_migration.py
   ```

4. **Update backend**:
   - Update Cloud Run environment variables
   - Deploy new version with AWS RDS URL

---

## 📞 Quick Commands Reference

```bash
# Check your current IP
curl ifconfig.me

# Test RDS connection
python3 test_aws_connection.py

# Run migration (when connection works)
python3 migrate_render_to_aws.py

# Check migration logs
cat migration.log
```

---

## 🎯 Ready to Go!

**Next action**: Add the security group rule using Option 1 above, then test!

The direct link again: https://console.aws.amazon.com/ec2/v2/home?region=eu-central-1#SecurityGroup:securityGroupId=sg-097205416c4b1f1e9
