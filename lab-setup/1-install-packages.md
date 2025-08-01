# JupyterHub + AI Environment Setup Document

---

## 1. System Update & Install Essentials

```bash
sudo apt update && sudo apt upgrade -y

# 1. Backup and replace sources.list with a fast mirror (Azure)
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak

sudo bash -c 'cat > /etc/apt/sources.list' <<EOF
deb http://azure.archive.ubuntu.com/ubuntu noble main restricted universe multiverse
deb http://azure.archive.ubuntu.com/ubuntu noble-updates main restricted universe multiverse
deb http://azure.archive.ubuntu.com/ubuntu noble-backports main restricted universe multiverse
deb http://security.ubuntu.com/ubuntu noble-security main restricted universe multiverse
EOF

# 2. Clean and refresh apt
sudo rm -rf /var/lib/apt/lists/*
sudo apt clean
sudo apt update

sudo apt install -y python3 python3-pip python3-dev python3.12-venv nodejs npm git
```

---

## 2. Install Configurable HTTP Proxy (required by JupyterHub)

```bash
sudo npm install -g configurable-http-proxy
```

---

## 3. Create and Activate Python Virtual Environment for AI & JupyterHub

```bash
# Create virtual environment (you can change path if needed)
sudo su
cd
sudo mkdir -p /opt/venvs/jupyterhub
sudo chown root:users /opt/venvs/jupyterhub
sudo chmod -R 2777 /opt/venvs/jupyterhub
export venv_name=venv_d1
python3 -m venv /opt/venvs/jupyterhub/$venv_name
source /opt/venvs/jupyterhub/$venv_name/bin/activate
```

---

## 4. Install AI and JupyterHub Python Packages

```bash
pip install --upgrade pip setuptools wheel

# JupyterHub ecosystem
pip install jupyterhub notebook jupyterlab jupyter-ai jupyterhub-idle-culler
```

---

## 5. Create User Accounts for Testing (u1 to u40)

```bash
for i in $(seq 1 9); do
    sudo useradd -m u0$i
    echo "u0$i:GeAI123456" | sudo chpasswd
done

for i in $(seq 10 40); do
    sudo useradd -m u$i
    echo "u$i:GeAI123456" | sudo chpasswd
done
```

---
