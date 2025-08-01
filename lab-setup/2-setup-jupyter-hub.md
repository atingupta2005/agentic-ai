# JupyterHub Setup

### 2. Create JupyterHub Configuration Directory and File

```bash
sudo su
sudo mkdir -p /etc/jupyterhub
sudo nano /etc/jupyterhub/jupyterhub_config.py
```

Paste this into `jupyterhub_config.py`:

```python
import pwd

all_users = {user.pw_name for user in pwd.getpwall()}

c.JupyterHub.authenticator_class = 'jupyterhub.auth.PAMAuthenticator'

# Allow only these 2 users
c.Authenticator.allowed_users = all_users

# Make both admins
c.Authenticator.admin_users = all_users

# Set spawner command to your venv's jupyterhub-singleuser
c.Spawner.cmd = ['/opt/venvs/jupyterhub/venv/bin/jupyterhub-singleuser']

# Bind to all IPs (optional)
c.JupyterHub.bind_url = 'http://0.0.0.0:8000'

# Enable debug logging
c.Application.log_level = 'DEBUG'
```

Save and exit.

---

### 3. Setup Systemd Service for JupyterHub

Create service file:

```bash
sudo nano /etc/systemd/system/jupyterhub.service
```

Paste:

```ini
[Unit]
Description=JupyterHub
After=network.target

[Service]
User=root
Environment="PATH=/opt/venvs/jupyterhub/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
ExecStart=/opt/venvs/jupyterhub/venv/bin/jupyterhub -f /etc/jupyterhub/jupyterhub_config.py

[Install]
WantedBy=multi-user.target
```

Save and exit.

---

### 4. Enable and Start JupyterHub Service

```bash
sudo su
sudo systemctl daemon-reload
sudo systemctl enable jupyterhub
sudo systemctl start jupyterhub
sudo systemctl status jupyterhub  # Verify service is running
```

---

### 5. Permissions (if needed)

Make sure users have execute permission on virtualenv folders:

```bash
sudo chmod -R o+rx /opt/venvs/jupyterhub/venv
```

---

### 6. Access

Open your browser and go to:

```
http://<server-ip>:8000
```

Log in with any OS user — all will have admin rights.

---

# Notes

* **Security risk:** This setup grants admin rights to all OS users. Use cautiously.
* Modify `/etc/jupyterhub/jupyterhub_config.py` if you want to restrict users or customize further.
* Monitor logs with:

```bash
sudo journalctl -u jupyterhub -f
```
