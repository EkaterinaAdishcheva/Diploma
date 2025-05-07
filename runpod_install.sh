cd /workspace/

mkdir /workspace/.ssh
ssh-keygen -f /workspace/.ssh/id_rsa
cp /workspace/.ssh/* /root/.ssh/
chmod 400 /root/.ssh/id_rsa.pub
chmod 400 /root/.ssh/id_rsa

git config --global user.email "e***a@example.com"
git config --global user.name "Ekaterina Adishcheva"

export HF_HOME="/workspace/.cache/"
export WANDB_API_KEY=16c91bc3767f14563f500e1342430f71c8b26518
cat .ssh/id_rsa.pub
