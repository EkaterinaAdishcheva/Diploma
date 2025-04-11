vcd /workspace/
python -m venv oa_venv
. oa_venv/bin/activate
python -m pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name=oa_venv --display-name "Python (oa_venv)"

mkdir /workspace/.ssh
ssh-keygen -f /workspace/.ssh/id_rsa
cp /workspace/.ssh/* /root/.ssh/
chmod 400 /root/.ssh/id_rsa.pub
chmod 400 /root/.ssh/id_rsa

git config --global user.email "you@example.com"
git config --global user.name "Your Name"

cat .ssh/id_rsa.pub


#One Actor
git clone git@github.com:EkaterinaAdishcheva/OneActor.git
pip install -r OneActor/requirements.txt
pip install -r OneActor/requirements_add.txt
deactivate oa_venv

#DreamSim
cd /workspace/
python -m venv ds
. ds/bin/activate
pip install ipykernel
python -m ipykernel install --user --name=ds --display-name "Python (ds)"
git clone https://github.com/ssundaram21/dreamsim.git
pip install -r dreamsim/requirements.txt
cp OneActor/ds_metrics.ipynb dreamsim

pip install torch
pip install Pillow
deactivate ds

sam2
https://github.com/facebookresearch/sam2/blob/main/notebooks/automatic_mask_generator_example.ipynb
? https://github.com/IDEA-Research/GroundingDINO/

deactivate ds
python -m venv sam2_venv
. sam2_venv/bin/activate
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

https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb#scrollTo=aCH4p1dtyaXX
