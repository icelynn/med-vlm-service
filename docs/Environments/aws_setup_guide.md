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

**Ollama port (11434) — do NOT open it; no app-level auth needed**
Decision (2026-06-20): the FastAPI proxy (`python/src/`) and Ollama run on the **same** EC2 instance, so the proxy reaches Ollama via `127.0.0.1:11434`. Ollama's own API has no authentication mechanism — security comes entirely from **not being reachable from outside the instance**, not from a token.

```bash
# 1. Confirm Ollama is installed via the standard script (binds 127.0.0.1:11434 by default,
#    no OLLAMA_HOST override needed/wanted)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Verify it's listening on localhost only — must show 127.0.0.1:11434, NOT 0.0.0.0:11434
ss -tlnp | grep 11434

# 3. Do NOT run an authorize-security-group-ingress for port 11434.
#    Only port 8000 (the FastAPI proxy, step above) should ever be opened.
#    If a prior session opened 11434, revoke it:
aws ec2 revoke-security-group-ingress \
    --group-id "$CURRENT_SG" \
    --protocol tcp \
    --port 11434 \
    --cidr "0.0.0.0/0"
```

If `ss` shows `0.0.0.0:11434` instead of `127.0.0.1:11434`, something set `OLLAMA_HOST=0.0.0.0` (e.g. a copy-pasted Docker tutorial) — fix it:
```bash
sudo systemctl edit ollama.service   # remove any Environment="OLLAMA_HOST=..." override
sudo systemctl restart ollama
ss -tlnp | grep 11434                # re-check: should now be 127.0.0.1:11434
```

**Post-install verification checklist (run once, right after Ollama is installed)**

| # | Check | Command (run on the EC2 instance unless noted) | Pass criteria |
|---|---|---|---|
| 1 | Service is running | `sudo systemctl status ollama` | `active (running)` |
| 2 | Bound to localhost only | `ss -tlnp \| grep 11434` | Shows `127.0.0.1:11434`, **not** `0.0.0.0:11434` |
| 3 | Reachable from the instance itself | `curl http://127.0.0.1:11434/api/tags` | Returns JSON (model list, possibly empty) |
| 4 | **Not** reachable from outside | From your **local machine** (not EC2): `curl --max-time 5 http://<EC2-public-IP>:11434/api/tags` | Times out / connection refused — if this succeeds, the port is exposed and something is wrong |
| 5 | Security group still has no 11434 rule | AWS Console → EC2 → Security Groups → Inbound rules (or `aws ec2 describe-security-groups --group-ids "$CURRENT_SG"`) | Only ports 22 and 8000 listed, 11434 absent |
| 6 | Required models are pulled | `ollama list` | Shows `qwen3-vl:4b` and `medgemma:4b` (or whichever `OLLAMA_MAIN_MODEL`/`OLLAMA_MEDICAL_MODEL` are set to) |
| 7 | The FastAPI proxy can reach Ollama end-to-end | With `ENV=demo` set and the service running: `curl -X POST http://localhost:8000/analyze -F "prompt=test" -F "image=@data/ct_ich/images/050_016.png;type=image/png"` | Returns `{"report": "..."}`, not a connection error |

If check #4 unexpectedly succeeds, stop and fix the security group / Ollama bind **before** doing anything else — that means the demo backend is open to the public internet with no authentication.

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

**Shell auto-logout on inactivity (TMOUT)**
If a user leaves an SSH session open without typing, the instance should eventually stop to save cost. `TMOUT` is a bash built-in that auto-logs out the shell after a period of keyboard inactivity:
```bash
sudo nano /etc/profile.d/autologout.sh
```
Write the following and save:
```bash
# Auto-logout after 30 minutes of shell inactivity
TMOUT=1800
readonly TMOUT
export TMOUT
```
```bash
sudo chmod +x /etc/profile.d/autologout.sh
```
`readonly` prevents the user from overriding the value. The setting takes effect on the **next SSH login** (or run `source /etc/profile.d/autologout.sh` to apply immediately).

> **Note:** `TMOUT` only applies to interactive bash. Full-screen programs (`vim`, `top`, `htop`) are unaffected — you will not be kicked out while editing a file.

**Linux cron activity heartbeat (prevents an accidental stop while you are actively working)**
Text-only operations (`nano`, `cat`) consume almost no CPU, so CloudWatch can misread them as idle. The following job raises CPU only when a user is **actively typing** (idle < 1 minute), preventing false stops during work while still allowing auto-stop when the user walks away:
```bash
sudo crontab -e
# Append to the bottom of the file (choose the nano editor with 1 on first open):
* * * * * for tty in $(who | awk '/pts/{print $2}'); do idle=$(w -h "$tty" 2>/dev/null | awk '{print $5}'); if [ "$idle" = "." ] || echo "$idle" | grep -qE '^[0-9]+(\.[0-9]+)?s$'; then openssl speed rsa1024 >/dev/null 2>&1; break; fi; done
```
Logic: the cron checks each SSH session's idle time via `w`. If any session has been idle for less than 1 minute (actively typing), it runs `openssl` to raise CPU and reset the CloudWatch timer. If all sessions are idle ≥ 1 minute, no heartbeat is sent — `TMOUT` will log out the shell after 30 minutes, then CloudWatch will stop the instance after another 30 minutes.

**Combined behavior summary:**

| Scenario | Heartbeat | TMOUT | CloudWatch alarm |
|---|---|---|---|
| User actively typing | ✅ fires every minute | Timer resets | Does not trigger |
| User idle < 30 min | ❌ stops firing | Timer counting | Does not trigger (within 30 min) |
| User idle ≥ 30 min | ❌ | Shell auto-logs out | Triggers after 30 more min → Stop |
| User forgot to log out and went to sleep | ❌ | Shell auto-logs out after 30 min | Triggers after 30 more min → Stop |

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
