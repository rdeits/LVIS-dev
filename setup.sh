DIRECTORY=$(cd `dirname "$0"` && pwd)
export JULIA_PKGDIR="$DIRECTORY/packages"
export PYTHON=""
echo "Julia package directory set to: $JULIA_PKGDIR"
export JULIA_LOAD_PATH="$JULIA_LOAD_PATH:$DIRECTORY/modules"
export JULIA_LOAD_PATH="$JULIA_LOAD_PATH:$DIRECTORY/submodules"
