export venv_name=venv_d3

python3 -m venv /opt/venvs/jupyterhub/$venv_name
source /opt/venvs/jupyterhub/$venv_name/bin/activate
pip install --upgrade pip setuptools wheel
pip install jupyterhub notebook jupyterlab jupyter-ai jupyterhub-idle-culler


sudo cp /etc/jupyterhub/jupyterhub_config_d0.py /etc/jupyterhub/jupyterhub_config_d3.py
# Change at 4 places: 3 ports and 1 path
sudo nano /etc/jupyterhub/jupyterhub_config_d3.py
sudo cat /etc/jupyterhub/jupyterhub_config_d3.py | grep url
sudo cat /etc/jupyterhub/jupyterhub_config_d3.py | grep venv_


sudo cp /etc/systemd/system/jupyterhub_d0.service /etc/systemd/system/jupyterhub_d3.service

# Change 3 places
sudo nano /etc/systemd/system/jupyterhub_d3.service


sudo su
sudo systemctl daemon-reload
sudo systemctl enable jupyterhub_d3
sudo systemctl stop jupyterhub_d3
sudo systemctl start jupyterhub_d3
sudo systemctl status jupyterhub_d3

sudo journalctl -u jupyterhub_d3.service -f --no-pager

curl localhost:8003

