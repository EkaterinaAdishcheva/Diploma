## fast run
cd /workspace/

cp /workspace/.ssh/* /root/.ssh/
chmod 400 /root/.ssh/id_rsa.pub
chmod 400 /root/.ssh/id_rsa

git config --global user.email "e***a@example.com"
git config --global user.name "Ekaterina Adishcheva"

. ds/bin/activate
python -m ipykernel install --user --name=ds --display-name "Python (ds)"
deactivate ds

. sam2_venv/bin/activate
python -m ipykernel install --user --name=sam2_venv --display-name "Python (sam2_venv)"
deactivate sam2_venv

. oa_venv/bin/activate
python -m ipykernel install --user --name=oa_venv --display-name "Python (oa_venv)"
cd OneActor
