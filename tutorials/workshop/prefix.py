# simple automated script for proper port forwarding of Solara app when launched inside Notebooks Hub VSCode instance
import os

# extract jupyterhub service prefix and modify for solara service prefix
jhsp = os.environ['JUPYTERHUB_SERVICE_PREFIX']
ssp = jhsp + 'proxy/8765'

# output the desired path to use
print(ssp)