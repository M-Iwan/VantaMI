# Download docker image
docker pull dptechnology/unimol:latest-pytorch1.11.0-cuda11.3

# Clone repository
git clone https://github.com/dptech-corp/Uni-pKa.git ~/.vantami/data/Uni-pKa

# Make the required Dockerfile
cat > ~/.vantami/data/Dockerfile <<'EOF'
FROM dptechnology/unimol:latest-pytorch1.11.0-cuda11.3

WORKDIR /opt/uni-pka

COPY Uni-pKa/ /opt/uni-pka/

ENV PYTHONPATH=/opt/uni-pka:${PYTHONPATH}
ENV PYTHONUNBUFFERED=1

RUN python --version \
    && python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda)" \
    && python -c "import rdkit; print('rdkit:', rdkit.__version__)" \
    && python -c "import unicore; print('unicore import: OK')"

ENTRYPOINT []
CMD ["python", "--version"]

EOF

# Build the docker image
docker build --no-cache --progress=plain -t uni-pka:local ~/.vantami/data

# Download the checkpoints for pKa model
hf download bindresearch/pkasso_unipka --local-dir ~/.vantami/data/Uni-pKa/models --include "fold_*/checkpoint_best.pt"