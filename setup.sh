DIRECTORY=$(cd `dirname "$0"` && pwd)
export JULIA_PKGDIR="$DIRECTORY/packages"
export PYTHON=""
echo "Julia package directory set to: $JULIA_PKGDIR"
