cd /workspace/
python -m venv sam2_venv
. sam2_venv/bin/activate
python -m pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name=sam2_venv --display-name "Python (sam2_venv)"
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .

cd checkpoints && \
./download_ckpts.sh && \
cd ..
pip install matplotlib
pip install opencv-python

deactivate sam2_venv

cp Diploma/sam2/automatic_mask_generator_example.ipynb sam2 