#DreamSim
cd /workspace/
python -m venv ds
. ds/bin/activate
python -m pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name=ds --display-name "Python (ds)"
git clone https://github.com/ssundaram21/dreamsim.git
pip install -r dreamsim/requirements.txt
pip install torch
pip install Pillow
pip install seaborn
deactivate ds

