# Source me:  source scripts/env.sh
#
# Puts the project's python env on PATH so scripts run as `python foo.py`
# instead of spelling out the interpreter each time, and adds the repo to
# PYTHONPATH. Mirrors fc_methods_comparison_cml/scripts/env.sh.
#
# The env comes from config.yaml's `python_env_path`, so this file and the
# config cannot drift apart. That key is `null` by default, meaning "use
# whatever env is already active" -- in which case this script only sets
# PYTHONPATH and leaves your interpreter alone.
#
# Parsing deliberately avoids importing yaml: this runs BEFORE the project env
# is on PATH, so it can only rely on the system python3, which may not have
# PyYAML. A grep for the single top-level key is enough.
_REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
_ENV=$(sed -n 's/^python_env_path:[[:space:]]*//p' "$_REPO/config/config.yaml" \
       | head -1 | tr -d '"'"'" | sed 's/[[:space:]]*$//')
_ENV=${_ENV/\$\{REPO_ROOT\}/$_REPO}
_ENV=${_ENV/\$\{USER\}/$USER}

if [ -z "$_ENV" ] || [ "$_ENV" = "null" ] || [ "$_ENV" = "~" ]; then
    # Default: no env pinned. Use whatever python is already active.
    export PYTHONPATH="$_REPO:${PYTHONPATH}"
    if command -v python >/dev/null 2>&1; then
        echo "env.sh: python_env_path is null -> using the active env: $(command -v python)"
    else
        echo "env.sh: python_env_path is null and no 'python' on PATH -- activate an env first" >&2
    fi
elif [ ! -x "$_ENV/bin/python" ]; then
    echo "env.sh: no interpreter at $_ENV/bin/python (check python_env_path in config/config.yaml)" >&2
else
    export PATH="$_ENV/bin:$PATH"
    export PYTHON="$_ENV/bin/python"
    export PYTHONPATH="$_REPO:${PYTHONPATH}"
    echo "env.sh: python -> $(command -v python)"
fi
unset _REPO _ENV
