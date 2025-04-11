cd /workspace/
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
https://github.com/facebookresearch/sam2/blob/main/notebooks/image_predictor_example.ipynb
https://github.com/facebookresearch/sam2/blob/main/notebooks/automatic_mask_generator_example.ipynb
https://github.com/IDEA-Research/GroundingDINO/
https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb#scrollTo=aCH4p1dtyaXX

python -m venv sam2
. sam2/bin/activate
pip install ipykernel
python -m ipykernel install --user --name=sam2 --display-name "Python (sam2)"
