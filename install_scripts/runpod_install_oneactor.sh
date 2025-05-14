#One Actor
cd /workspace/
git clone git@github.com:EkaterinaAdishcheva/Diploma.git
python -m venv oa_venv
. oa_venv/bin/activate
python -m pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name=oa_venv --display-name "Python (oa_venv)"

pip install -r Diploma/requirements.txt
pip install -r Diploma/requirements_tg_bot.txt

export HF_HOME="/workspace/.cache/"
deactivate oa_venv
