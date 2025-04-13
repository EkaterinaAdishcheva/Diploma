cd /workspace/

mkdir /workspace/.ssh
ssh-keygen -f /workspace/.ssh/id_rsa
cp /workspace/.ssh/* /root/.ssh/
chmod 400 /root/.ssh/id_rsa.pub
chmod 400 /root/.ssh/id_rsa

git config --global user.email "e***a@example.com"
git config --global user.name "Ekaterina Adishcheva"

cat .ssh/id_rsa.pub