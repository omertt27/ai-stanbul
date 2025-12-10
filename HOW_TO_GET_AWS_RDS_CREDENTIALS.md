# 🔐 How to Get AWS RDS Credentials

## Your RDS Instance Details

- **Instance ID**: database-1
- **Endpoint**: database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com
- **Port**: 5432
- **Region**: eu-central-1 (Frankfurt)
- **Database Name**: postgres (default)

---

## 🔑 Getting Master Username and Password

### Where to Find Them:

#### Option 1: AWS Console (Configuration Tab)

```
1. Go to AWS Console: https://console.aws.amazon.com/rds/
2. Navigate to "Databases"
3. Click on "database-1"
4. Go to "Configuration" tab
5. Look for "Master username" - this is your username
```

**Note**: The password is NOT shown in the console (for security).

#### Option 2: AWS Secrets Manager (if enabled)

```
1. Go to AWS Console: https://console.aws.amazon.com/secretsmanager/
2. Look for secret named "rds-db-credentials" or similar
3. Click "Retrieve secret value"
4. Copy username and password
```

---

## 🔍 What Username Did You Use?

When you created the RDS instance, you were asked to set:

1. **Master username** - Common defaults:
   - `postgres` (most common)
   - `admin`
   - `root`
   - Custom name you chose

2. **Master password** - You set this during creation
   - Minimum 8 characters
   - Must contain letters and numbers

---

## ❓ I Forgot My Password!

### If you don't remember the password, you can reset it:

#### Via AWS Console:

```
1. Go to AWS Console → RDS → Databases
2. Click on "database-1"
3. Click "Modify" button (top right)
4. Scroll to "Settings" section
5. Check "New master password"
6. Enter new password (twice)
7. Click "Continue"
8. Select "Apply immediately"
9. Click "Modify DB instance"
```

**⏱️ Takes 2-5 minutes to apply**

#### Via AWS CLI:

```bash
aws rds modify-db-instance \
  --db-instance-identifier database-1 \
  --master-user-password YOUR_NEW_PASSWORD \
  --apply-immediately \
  --region eu-central-1
```

---

## 🧪 Test Connection with Different Usernames

Try common usernames:

```bash
# Test with 'postgres' (most common)
psql "postgresql://postgres:YOUR_PASSWORD@database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com:5432/postgres"

# Test with 'admin'
psql "postgresql://admin:YOUR_PASSWORD@database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com:5432/postgres"

# Or use the test script
cd /Users/omer/Desktop/ai-stanbul
python3 test_rds_connection.py
# Enter username when prompted
```

---

## 📋 Common Usernames to Try

1. **postgres** (PostgreSQL default)
2. **admin** (AWS common default)
3. **root** (MySQL-style, less common)
4. **aiistanbul_admin** (if you followed our guide)
5. **Your custom username** (if you set one)

---

## 🔧 Check Current Configuration

### Via AWS CLI:

```bash
# Get RDS instance details
aws rds describe-db-instances \
  --db-instance-identifier database-1 \
  --region eu-central-1 \
  --query 'DBInstances[0].[MasterUsername,DBName,Endpoint.Address]' \
  --output table
```

This will show:
- Master username
- Database name
- Endpoint address

---

## ✅ Once You Have Credentials

Update your `.env` file:

```bash
cd /Users/omer/Desktop/ai-stanbul

# Edit .env
nano .env

# Update this line:
DATABASE_URL=postgresql://YOUR_USERNAME:YOUR_PASSWORD@database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com:5432/postgres

# Example with username 'postgres' and password 'mypass123':
DATABASE_URL=postgresql://postgres:mypass123@database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com:5432/postgres
```

---

## 🧪 Test Connection

```bash
# Method 1: Using psql
psql "postgresql://USERNAME:PASSWORD@database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com:5432/postgres"

# Method 2: Using test script
cd /Users/omer/Desktop/ai-stanbul
python3 test_rds_connection.py
# Enter username and password when prompted

# Method 3: Using Python
python3 -c "
import psycopg2
conn = psycopg2.connect(
    host='database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com',
    port=5432,
    database='postgres',
    user='YOUR_USERNAME',
    password='YOUR_PASSWORD'
)
print('✅ Connected!')
conn.close()
"
```

---

## 🆘 Still Can't Connect?

### Check These:

1. **Is RDS publicly accessible?**
   ```
   AWS Console → RDS → database-1 → Connectivity & security
   Look for "Publicly accessible: Yes"
   ```

2. **Security group allows port 5432?**
   ```
   AWS Console → EC2 → Security Groups → sg-097205416c4b1f1e9
   Inbound rules should have: PostgreSQL (5432) from 0.0.0.0/0
   ```

3. **Correct region?**
   ```
   Make sure you're in eu-central-1 (Frankfurt)
   ```

4. **Username/password correct?**
   ```
   Try resetting password via AWS Console
   ```

---

## 📝 Your Current Setup

Based on your files, you need to:

1. **Find or reset AWS RDS password**
2. **Update `.env` file** with:
   ```
   RENDER_DATABASE_URL=postgresql://aistanbul_postgre_user:FEddnYmd0ymR2HKBJIax3mqWkfTB0XZe@dpg-d4jg45e3jp1c73b6gas0-a.frankfurt-postgres.render.com/aistanbul_postgre
   
   DATABASE_URL=postgresql://YOUR_USERNAME:YOUR_PASSWORD@database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com:5432/postgres
   ```

3. **Test connection**:
   ```bash
   python3 test_rds_connection.py
   ```

4. **Run migration**:
   ```bash
   python migrate_render_to_aws.py
   ```

---

## 🎯 Quick Reset & Test

```bash
# 1. Reset password in AWS Console (takes 3 minutes)
# 2. Update .env
cd /Users/omer/Desktop/ai-stanbul
nano .env

# 3. Test
python3 test_rds_connection.py

# 4. If successful, migrate
python migrate_render_to_aws.py
```

---

## 📞 Need Help?

Common issues:
- **"Connection timeout"** → Check security group
- **"Password authentication failed"** → Wrong password, reset it
- **"Could not resolve hostname"** → Check endpoint URL
- **"No pg_hba.conf entry"** → Security group or RDS not public

---

**Next Step**: Get or reset your AWS RDS password, then update the `.env` file!
