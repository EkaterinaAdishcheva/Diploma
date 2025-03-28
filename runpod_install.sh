python -m venv oa_venv
. oa_venv/bin/activate
pip install ipykernel
python -m ipykernel install --user --name=oa_venv --display-name "Python (oa_venv)"

mkdir /workspace/.ssh
ssh-keygen -f /workspace/.ssh/id_rsa
cp /workspace/.ssh/* /root/.ssh/
chmod 400 /root/.ssh/id_rsa.pub
chmod 400 /root/.ssh/id_rsa

pip install -r requirements.txt

git config --global user.email "you@example.com"
git config --global user.name "Your Name"