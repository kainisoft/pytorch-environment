# Jupyter Server Configuration
# This configuration file disables authentication completely

c = get_config()

# Disable token authentication
c.ServerApp.token = ''
c.ServerApp.password = ''

# Disable XSRF protection for local development
c.ServerApp.disable_check_xsrf = True

# Allow all origins (for local development)
c.ServerApp.allow_origin = '*'

# Allow remote access
c.ServerApp.allow_remote_access = True

# Disable authentication for Prometheus metrics
c.ServerApp.authenticate_prometheus = False

# Don't open browser automatically
c.ServerApp.open_browser = False

# Set IP and port
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888

# Allow root user
c.ServerApp.allow_root = True

# Disable password requirement
c.ServerApp.password_required = False

# Additional security settings for local development
c.ServerApp.trust_xheaders = True
