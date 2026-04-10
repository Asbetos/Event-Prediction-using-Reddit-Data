#!/usr/bin/env bash
set -euo pipefail

echo "=== Reddit Early Warning System: EC2 Environment Setup ==="

# 1. JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
if ! grep -q 'JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64' ~/.bashrc; then
    echo 'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64' >> ~/.bashrc
    echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.bashrc
fi
echo "JAVA_HOME set to $JAVA_HOME"

# 2. Python dependencies
pip3 install --user --break-system-packages -q \
    'pyspark==3.5.5' \
    'pandas>=2.0' \
    'numpy>=1.24' \
    'pyarrow>=14.0' \
    'matplotlib>=3.8' \
    'plotly>=5.18' \
    'kaleido>=0.2.1' \
    'scikit-learn>=1.3' \
    'scipy>=1.11' \
    'networkx>=3.1' \
    'wordcloud>=1.9' \
    'tqdm>=4.66' \
    'boto3>=1.34' \
    's3fs>=2024.2.0'

echo "Python packages installed"

# 3. Quarto
if ! command -v quarto &> /dev/null; then
    QUARTO_VERSION="1.4.557"
    wget -q "https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-amd64.deb" \
        -O /tmp/quarto.deb
    sudo dpkg -i /tmp/quarto.deb
    rm /tmp/quarto.deb
    echo "Quarto ${QUARTO_VERSION} installed"
else
    echo "Quarto already installed: $(quarto --version)"
fi

# 4. System fonts for matplotlib
sudo apt-get update -qq
sudo apt-get install -y -qq libfontconfig1 fonts-dejavu-core 2>/dev/null || true

echo "=== EC2 setup complete ==="
