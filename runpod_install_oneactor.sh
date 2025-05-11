#One Actor
cd /workspace/
git clone git@github.com:EkaterinaAdishcheva/Diploma.git
python -m venv oa_venv
. oa_venv/bin/activate
python -m pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name=oa_venv --display-name "Python (oa_venv)"
pip install -r Diploma/OneActor/requirements.txt
export HF_HOME="/workspace/.cache/"
deactivate oa_venv
