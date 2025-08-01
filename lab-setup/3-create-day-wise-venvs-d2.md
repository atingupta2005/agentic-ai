export venv_name=venv_d2

python3 -m venv /opt/venvs/jupyterhub/$venv_name
source /opt/venvs/jupyterhub/$venv_name/bin/activate
pip install --upgrade pip setuptools wheel
pip install jupyterhub notebook jupyterlab jupyter-ai jupyterhub-idle-culler


sudo cp /etc/jupyterhub/jupyterhub_config_d0.py /etc/jupyterhub/jupyterhub_config_d2.py
# Change at 4 places: 3 ports and 1 path
sudo nano /etc/jupyterhub/jupyterhub_config_d2.py
sudo cat /etc/jupyterhub/jupyterhub_config_d2.py | grep url
sudo cat /etc/jupyterhub/jupyterhub_config_d2.py | grep venv_


sudo cp /etc/systemd/system/jupyterhub_d0.service /etc/systemd/system/jupyterhub_d2.service
sudo nano /etc/systemd/system/jupyterhub_d2.service


sudo su
sudo systemctl daemon-reload
sudo systemctl enable jupyterhub_d2
sudo systemctl stop jupyterhub_d2
sudo systemctl start jupyterhub_d2
sudo systemctl status jupyterhub_d2

sudo journalctl -u jupyterhub_d2.service -f --no-pager

curl localhost:8002

