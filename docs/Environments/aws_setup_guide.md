# AWS Setup Guide

Covers EC2 instance provisioning, SSH access, security-group ingress, and the CloudWatch idle auto-stop mechanism.
Back to overview: [Environment & Dependency Overview](./environment_and_dependencies_overview.md)

## 1. Setup

**Provision the EC2 instance**
Following the [cloud hardware specification](./environment_and_dependencies_overview.md#2-cloud-hardware-specification), launch a `g4dn.xlarge` from the AWS console with the Ubuntu 22.04 LTS AMI and a root volume ≥ 50 GB.

**Connect via SSH and update the system**
```bash
ssh -i "your-key.pem" ubuntu@your-ec2-ip

sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y build-essential curl git python3-pip python3-venv
```

**Open port 8000 (one-shot hot update from your local machine, no SSH required)**
```bash
# 1. Resolve the security-group ID currently bound to the instance
CURRENT_SG=$(aws ec2 describe-instances \
    --instance-ids "your-instance-id" \
    --query "Reservations[0].Instances[0].SecurityGroups[0].GroupId" \
    --output text)

# 2. Add an inbound rule for port 8000 (0.0.0.0/0 allows HTTP traffic from anywhere)
aws ec2 authorize-security-group-ingress \
    --group-id "$CURRENT_SG" \
    --protocol tcp \
    --port 8000 \
    --cidr "0.0.0.0/0"
```

**CloudWatch idle auto-stop (cost control)**
Monitor `CPUUtilization` to detect true idleness. After 30 continuous minutes below 2%, the instance auto-stops (billing halts; the EBS volume is retained).

Console configuration:
1. EC2 Console → select the instance → **Monitoring** tab.
2. On the "CPU utilization (Percent)" chart, click the ⋮ menu → **Create alarm**.
3. Metric and conditions: Statistic = `Average`, Period = `5 minutes`, condition = `Lower/Equal`, threshold = `2`.
4. Advanced configuration: set `Datapoints to alarm` to **6 out of 6** (5 min × 6 = 30 continuous idle minutes).
5. Alarm action: remove the SNS notification → **Add EC2 action** → In alarm → **Stop this instance**.
   - Select `Stop`, never `Terminate` (Terminate permanently destroys the instance and wipes all data).
6. Name the alarm `med-vlm-gpu-idle-30min-stop` → Create alarm.

**Linux cron activity heartbeat (prevents an accidental stop while you are working)**
Text-only operations (`nano`, `cat`) consume almost no CPU, so CloudWatch can misread them as idle. The following job keeps CPU above the threshold whenever a user is connected via SSH:
```bash
sudo crontab -e
# Append to the bottom of the file (choose the nano editor with 1 on first open):
* * * * * who | grep -q "pts" && openssl speed rsa1024 >/dev/null 2>&1
```
Logic: a user is online → openssl raises CPU → the CloudWatch timer resets (no stop); the user logs off → CPU drops to zero → the instance auto-stops after 30 minutes.

## 2. Troubleshooting

| Symptom | Root cause | Fix |
|---|---|---|
| `connect to host ... port 22: Connection timed out` | The security-group inbound rules do not include your current public IP, or contain a private IP (`192.168.x.x`) | Get your public IP with `curl http://ifconfig.me`, then `aws ec2 authorize-security-group-ingress ... --cidr "public-ip/32"`. Also confirm no local VPN is active and the network does not block port 22 |
| `Permission denied (publickey)` | (1) Corrupted key encoding (PowerShell `>` defaults to UTF-16; OpenSSH accepts only UTF-8); (2) key and instance out of sync (a key pair created after the instance launched cannot be recognized by the running instance) | `chmod 400 your-key.pem`. If it still fails, rebuild cleanly: delete the old cloud and local keys → regenerate as UTF-8 → launch a fresh instance with the new key |
| `Connection closed by ... port 22` (drops immediately after typing `yes` on first connect) | Not an error; the final step of `cloud-init` restarts `sshd`, which drops the in-progress session | Wait 60–90 seconds for initialization to finish, then press up-arrow and re-run `ssh` |

**Regenerate the key (UTF-8)**
```bash
aws ec2 delete-key-pair --key-name "old-key-name"
rm *.pem
aws ec2 create-key-pair --key-name "my-key" \
    --query "KeyMaterial" --output text > my-key.pem
# Then terminate the old instance and launch a fresh one with the new key
```

## 3. Verification

```bash
# Probe the network path (either command)
Test-NetConnection -ComputerName XX.XX.XX.XX -Port 22   # Windows
nc -zv XX.XX.XX.XX 22                                    # macOS / Linux
```
- A successful SSH login confirms both the network and authentication layers are open.
- The CloudWatch alarm transitions to `In alarm` and triggers a stop while idle, confirming the cost-control mechanism works.
