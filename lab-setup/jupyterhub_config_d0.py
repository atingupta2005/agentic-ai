import pwd

all_users = {user.pw_name for user in pwd.getpwall()}

c.JupyterHub.authenticator_class = 'jupyterhub.auth.PAMAuthenticator'

# Allow only these 2 users
c.Authenticator.allowed_users = all_users

# Make both admins
c.Authenticator.admin_users = all_users

# Set spawner command to your venv's jupyterhub-singleuser
c.Spawner.cmd = ['/opt/venvs/jupyterhub/venv_d0/bin/jupyterhub-singleuser']

# Bind to all IPs (optional)
c.JupyterHub.bind_url = 'http://0.0.0.0:8000'
c.JupyterHub.hub_bind_url = 'http://127.0.0.1:8080'
c.ConfigurableHTTPProxy.api_url = 'http://127.0.0.1:8010'

# Enable debug logging
c.Application.log_level = 'DEBUG'
