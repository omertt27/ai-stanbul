# AWS RDS PostgreSQL Connectivity Setup

## 🔴 CRITICAL ISSUE: Database Not Publicly Accessible

Your RDS instance `database-1` is configured as **NOT publicly accessible**, which means Google Cloud Run cannot connect to it directly.

### Current Configuration
- **Instance ID**: database-1
- **Endpoint**: database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com
- **Port**: 5432
- **Region**: eu-central-1 (Frankfurt)
- **Instance Class**: db.r7g.large (ARM-based Graviton)
- **Storage**: 500 GB GP3 SSD (3000 IOPS, 125 MB/s)
- **VPC**: vpc-0cbb00ea173822ceb
- **Security Group**: sg-097205416c4b1f1e9
- **Availability Zone**: eu-central-1a
- **Public Access**: ❌ **NO**

---

## Solutions (Choose One)

### ✅ Option 1: Enable Public Access (Fastest - Testing Only)

**⚠️ Not recommended for production, but quickest for testing**

#### Step 1: Modify RDS Instance
```bash
# Via AWS Console:
1. Open AWS Console → RDS → Databases
2. Select "database-1"
3. Click "Modify"
4. Scroll to "Connectivity"
5. Under "Additional configuration", set "Publicly accessible" to "Yes"
6. Click "Continue"
7. Select "Apply immediately"
8. Click "Modify DB instance"

# Or via AWS CLI:
aws rds modify-db-instance \
  --db-instance-identifier database-1 \
  --publicly-accessible \
  --apply-immediately \
  --region eu-central-1
```

#### Step 2: Update Security Group
```bash
# Add inbound rule to allow connections from anywhere (testing only)
aws ec2 authorize-security-group-ingress \
  --group-id sg-097205416c4b1f1e9 \
  --protocol tcp \
  --port 5432 \
  --cidr 0.0.0.0/0 \
  --region eu-central-1

# For production, restrict to specific IPs:
# --cidr <your-cloud-run-ip-range>/32
```

#### Step 3: Test Connection
```bash
# Test from your local machine
psql "postgresql://username:password@database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com:5432/postgres"

# Or using Python
python3 << EOF
import psycopg2
conn = psycopg2.connect(
    host='database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com',
    port=5432,
    database='postgres',
    user='your-username',
    password='your-password'
)
print("✅ Connected successfully!")
conn.close()
EOF
```

---

### 🔒 Option 2: VPC Peering (Production - Most Secure)

Connect Google Cloud Run to your AWS VPC using VPC peering.

#### Architecture
```
Google Cloud Run (GCP) 
    ↓ (VPC Connector)
Google Cloud VPC
    ↓ (VPC Peering)
AWS VPC (vpc-0cbb00ea173822ceb)
    ↓
RDS PostgreSQL (database-1)
```

#### Step 1: Create VPC Peering
```bash
# In AWS:
1. Go to VPC → Peering Connections → Create Peering Connection
2. Name: gcp-to-aws-peering
3. VPC (Requester): vpc-0cbb00ea173822ceb
4. Account: <GCP-account-id>
5. VPC (Accepter): <GCP-VPC-id>
6. Click "Create Peering Connection"

# In GCP:
gcloud compute networks peerings create aws-vpc-peering \
  --network=<your-gcp-vpc> \
  --peer-network=vpc-0cbb00ea173822ceb \
  --peer-project=<your-aws-account>
```

#### Step 2: Create Serverless VPC Connector
```bash
gcloud compute networks vpc-access connectors create aws-rds-connector \
  --network=<your-gcp-vpc> \
  --region=europe-west1 \
  --range=10.8.0.0/28

# Deploy Cloud Run with VPC connector
gcloud run deploy ai-istanbul-backend \
  --vpc-connector=aws-rds-connector \
  --vpc-egress=private-ranges-only
```

#### Step 3: Update Security Group
```bash
# Allow traffic from GCP VPC CIDR range
aws ec2 authorize-security-group-ingress \
  --group-id sg-097205416c4b1f1e9 \
  --protocol tcp \
  --port 5432 \
  --cidr <gcp-vpc-cidr>/16 \
  --region eu-central-1
```

---

### 🌉 Option 3: SSH Tunnel / Bastion Host

Use an EC2 instance as a jump host.

#### Step 1: Launch EC2 Bastion Host
```bash
# In AWS Console:
1. EC2 → Launch Instance
2. Name: rds-bastion
3. AMI: Amazon Linux 2023
4. Instance Type: t2.micro (free tier)
5. VPC: vpc-0cbb00ea173822ceb (same as RDS)
6. Subnet: Same availability zone as RDS (eu-central-1a)
7. Security Group: Allow SSH (22) from your IP
8. Create and download key pair
```

#### Step 2: Create SSH Tunnel
```bash
# From your local machine
ssh -i bastion-key.pem -L 5432:database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com:5432 ec2-user@<bastion-public-ip>

# In another terminal, connect to localhost:5432
psql "postgresql://username:password@localhost:5432/postgres"
```

#### Step 3: Connect Cloud Run via Bastion
```bash
# Not recommended - use VPC peering instead
# Bastion hosts are mainly for human access, not service-to-service
```

---

### 🔌 Option 4: AWS PrivateLink (Advanced)

Expose RDS via PrivateLink endpoint service to GCP.

#### Step 1: Create Network Load Balancer
```bash
# Create NLB targeting RDS endpoint
aws elbv2 create-load-balancer \
  --name rds-privatelink-nlb \
  --type network \
  --scheme internal \
  --subnets subnet-xxx subnet-yyy \
  --region eu-central-1
```

#### Step 2: Create VPC Endpoint Service
```bash
aws ec2 create-vpc-endpoint-service-configuration \
  --network-load-balancer-arns arn:aws:elasticloadbalancing:... \
  --acceptance-required
```

#### Step 3: Connect from GCP
```bash
# Create VPC endpoint in GCP side
# This requires complex cross-cloud networking setup
# Contact AWS/GCP support for guidance
```

---

## 🚀 Recommended Approach

### For Testing (Next 24 Hours)
**Use Option 1: Enable Public Access**
- Fastest setup (5 minutes)
- Allows immediate testing
- Can restrict to specific IPs

### For Production (After Testing)
**Use Option 2: VPC Peering**
- Most secure
- Better performance
- Standard cloud architecture
- Well-documented by both AWS and GCP

---

## 📝 Environment Variables

After choosing a connectivity option, update your `.env.production`:

```bash
# Database Configuration
DATABASE_URL=postgresql://username:password@database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com:5432/postgres
DB_HOST=database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com
DB_PORT=5432
DB_NAME=postgres
DB_USER=your-username
DB_PASSWORD=your-secure-password
DB_REGION=eu-central-1

# AWS Configuration
AWS_REGION=eu-central-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
```

---

## ✅ Verification Checklist

After setup, verify connectivity:

```bash
# 1. Test DNS resolution
nslookup database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com

# 2. Test port connectivity
nc -zv database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com 5432

# 3. Test PostgreSQL connection
psql "postgresql://username:password@database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com:5432/postgres" -c "SELECT version();"

# 4. Test from Cloud Run
# Deploy backend and check logs for connection errors
gcloud run logs read ai-istanbul-backend --limit=50
```

---

## 🆘 Troubleshooting

### Error: "could not connect to server"
```bash
# Check:
1. Security group allows traffic on port 5432
2. RDS is publicly accessible (if using Option 1)
3. Network ACLs allow traffic
4. Username/password are correct
```

### Error: "timeout"
```bash
# Check:
1. DNS resolves correctly
2. No firewall blocking port 5432
3. VPC routing configured (if using Option 2)
4. Cloud Run has VPC connector (if using Option 2)
```

### Error: "authentication failed"
```bash
# Check:
1. Username and password are correct
2. User has permissions on the database
3. Password doesn't contain special characters that need escaping
```

---

## 📚 Additional Resources

- [AWS RDS Security Groups](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Overview.RDSSecurityGroups.html)
- [GCP Serverless VPC Access](https://cloud.google.com/vpc/docs/configure-serverless-vpc-access)
- [VPC Peering Between GCP and AWS](https://cloud.google.com/vpc/docs/vpc-peering)
- [AWS PrivateLink](https://docs.aws.amazon.com/vpc/latest/privatelink/what-is-privatelink.html)

---

## 🎯 Quick Start Command

Enable public access now for testing:

```bash
# 1. Enable public access
aws rds modify-db-instance \
  --db-instance-identifier database-1 \
  --publicly-accessible \
  --apply-immediately \
  --region eu-central-1

# 2. Add security group rule
aws ec2 authorize-security-group-ingress \
  --group-id sg-097205416c4b1f1e9 \
  --protocol tcp \
  --port 5432 \
  --cidr 0.0.0.0/0 \
  --region eu-central-1

# 3. Wait 2-3 minutes for changes to apply
# 4. Test connection
psql "postgresql://username:password@database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com:5432/postgres"
```

---

**Status**: ⏳ Waiting for connectivity configuration
**Next Step**: Choose and implement one of the options above
