#!/usr/bin/env bash

INSTALL_CUDNN=${INSTALLCUDNN:-"true"}
INSTALL_NVTX=${INSTALLNVTX:-"true"}
INSTALL_NVCC=${INSTALLNVCC:-"true"}
CUDA_VERSION=${CUDAVERSION:-"11.7"}
CUDNN_VERSION=${CUDNNVERSION:-"8.5.0.96"}


# Add NVIDIA's package repository to apt so that we can download packages
# Always use the ubuntu2004 repo because the other repos (e.g., debian11) are missing packages
NVIDIA_REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64"
KEYRING_PACKAGE="cuda-keyring_1.0-1_all.deb"
KEYRING_PACKAGE_URL="$NVIDIA_REPO_URL/$KEYRING_PACKAGE"
KEYRING_PACKAGE_PATH="$(mktemp -d)"
KEYRING_PACKAGE_FILE="$KEYRING_PACKAGE_PATH/$KEYRING_PACKAGE"

# Ensure that the requested version of CUDA is available
CUDA_PKG="cuda-libraries-${CUDA_VERSION/./-}"
NVTX_PKG="cuda-nvtx-${CUDA_VERSION/./-}"
NVCC_PKG="cuda-nvcc-${CUDA_VERSION/./-}"
# Ensure that the requested version of cuDNN is available AND compatible
CUDNN_PKG_VERSION="libcudnn8=${CUDNN_VERSION}-1+cuda${CUDA_VERSION}"

set -e
export DEBIAN_FRONTEND=noninteractive

if [ "$(id -u)" -ne 0 ]; then
    echo -e 'Script must be run as root. Use sudo, su, or add "USER root" to your Dockerfile before running this script.'
    exit 1
fi

# Ensure that login shells get the correct path if the user updated the PATH using ENV.
rm -f /etc/profile.d/00-restore-env.sh
echo "export PATH=${PATH//$(sh -lc 'echo $PATH')/\$PATH}" > /etc/profile.d/00-restore-env.sh
chmod +x /etc/profile.d/00-restore-env.sh

# Checks if packages are installed and installs them if not
check_packages() {
    if ! dpkg -s "$@" > /dev/null 2>&1; then
        apt-get update -y
        apt-get -y install --no-install-recommends "$@"

        # Clean up
        apt-get clean -y
        rm -rf /var/lib/apt/lists/*
    fi
}

install_package() {
    if ! apt-cache show "$1"; then
        echo "The requested version of $1 is not available!"
        exit 1
    fi

    echo "(*) Installing "$1"..."
    apt-get update -y
    apt-get -y install --no-install-recommends "$1"

    # Clean up
    apt-get clean -y
    rm -rf /var/lib/apt/lists/*
}

# Install dependencies
echo "Installing dependencies..."
check_packages wget ca-certificates

wget -O "$KEYRING_PACKAGE_FILE" "$KEYRING_PACKAGE_URL"
apt-get -y install --no-install-recommends "$KEYRING_PACKAGE_FILE"
apt-get update -y

install_package "$CUDA_PKG"

if [ "$INSTALL_CUDNN" = "true" ]; then
    install_package $CUDNN_PKG_VERSION
fi

if [ "$INSTALL_NVTX" = "true" ]; then
    install_package $NVTX_PKG
fi

if [ "$INSTALL_NVCC" = "true" ]; then
    install_package $NVCC_PKG
fi

echo "Done!"