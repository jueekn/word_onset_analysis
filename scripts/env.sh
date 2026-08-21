# Source me:  source scripts/env.sh
#
# Puts the project's python env on PATH so scripts run as `python foo.py`
# instead of spelling out the interpreter each time. Mirrors
# fc_methods_comparison_cml/scripts/env.sh.
#
# The env is read from config.yaml's `python_env_path` so this file and the
# config cannot drift apart.
_REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
_ENV=$(python3 -c "
import sys,yaml,pathlib
cfg=yaml.safe_load(open('$_REPO/config/config.yaml'))
p=str(cfg['python_env_path']).replace('\${REPO_ROOT}','$_REPO')
print(p)")
if [ ! -x "$_ENV/bin/python" ]; then
    echo "env.sh: no interpreter at $_ENV/bin/python (check python_env_path in config/config.yaml)" >&2
else
    export PATH="$_ENV/bin:$PATH"
    export PYTHON="$_ENV/bin/python"
    export PYTHONPATH="$_REPO:${PYTHONPATH}"
    echo "env.sh: python -> $(command -v python)"
fi
unset _REPO _ENV
