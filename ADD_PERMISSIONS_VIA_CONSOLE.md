# ⚠️ IMPORTANT: Add Permissions Via AWS Console

## The Problem
Your IAM user `render-mvp-user` cannot add permissions to itself. This is normal AWS security behavior.

## ✅ SOLUTION: Use AWS Console (2 Minutes)

### Step-by-Step Instructions:

1. **Open this URL in your browser:**
   ```
   https://console.aws.amazon.com/iam/home#/users/render-mvp-user
   ```

2. **Log in with an admin account** (the account that created `render-mvp-user`)

3. **Click the "Permissions" tab**

4. **Click "Add permissions" button → Select "Attach policies directly"**

5. **Search and CHECK these 4 policies:**
   - ✅ `AmazonEC2ContainerRegistryFullAccess`
   - ✅ `IAMReadOnlyAccess`
   - ✅ `SecretsManagerReadWrite`
   - ✅ `AWSBatchFullAccess`

6. **Click "Next" → Click "Add permissions"**

7. **Done! Come back to terminal and run:**
   ```bash
   cd ~/Desktop/ai-stanbul
   ./push_to_ecr.sh
   ```

---

## What to Click (Visual Guide)

```
AWS Console → IAM → Users → render-mvp-user
    ↓
[Permissions] tab
    ↓
[Add permissions] button
    ↓
○ Add user to group
● Attach policies directly  ← SELECT THIS
○ Copy permissions
    ↓
Search box: type "ECR"
    ✅ AmazonEC2ContainerRegistryFullAccess
    ↓
Search box: type "IAM"
    ✅ IAMReadOnlyAccess
    ↓
Search box: type "Secrets"
    ✅ SecretsManagerReadWrite
    ↓
Search box: type "Batch"
    ✅ AWSBatchFullAccess
    ↓
[Next] button
    ↓
[Add permissions] button
    ↓
✅ DONE!
```

---

## Alternative: Ask AWS Admin

If you're not the admin, send this message to your AWS administrator:

```
Hi,

Can you please add these IAM policies to user "render-mvp-user"?

1. AmazonEC2ContainerRegistryFullAccess
2. IAMReadOnlyAccess
3. SecretsManagerReadWrite
4. AWSBatchFullAccess

This is needed to deploy our Docker containers to AWS ECR and Batch.

IAM Console Link:
https://console.aws.amazon.com/iam/home#/users/render-mvp-user

Thanks!
```

---

## After Adding Permissions

Test that it works:

```bash
# This should now succeed
aws ecr get-authorization-token --region eu-central-1

# Then push your Docker image
cd ~/Desktop/ai-stanbul
./push_to_ecr.sh
```

---

## Why This Happened

AWS security best practice: Users can't modify their own permissions. Only admins (users with IAM permissions) can add/remove permissions.

This prevents compromised credentials from escalating their own privileges.

---

## Quick Links

- **Add permissions (console):** https://console.aws.amazon.com/iam/home#/users/render-mvp-user?section=permissions
- **Your account login:** https://701893740767.signin.aws.amazon.com/console

---

**Next Step:** Log in to AWS Console with admin account → Add the 4 policies → Return here and run `./push_to_ecr.sh` 🚀
