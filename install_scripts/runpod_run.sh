## fast run
cd /workspace/

cp /workspace/.ssh/* /root/.ssh/
chmod 400 /root/.ssh/id_rsa.pub
chmod 400 /root/.ssh/id_rsa

git config --global user.email "your_email@example.com"
git config --global user.name "Your Name"

mkdir /workspace/.cache/
export HF_HOME="/workspace/.cache/"

. ds/bin/activate
python -m ipykernel install --user --name=ds --display-name "Python (ds)"
deactivate ds

. oa_venv/bin/activate
python -m ipykernel install --user --name=oa_venv --display-name "Python (oa_venv)"
