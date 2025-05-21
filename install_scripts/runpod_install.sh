cd /workspace/

mkdir /workspace/.ssh
ssh-keygen -f /workspace/.ssh/id_rsa
cp /workspace/.ssh/* /root/.ssh/
chmod 400 /root/.ssh/id_rsa.pub
chmod 400 /root/.ssh/id_rsa

git config --global user.email "your_email@example.com"
git config --global user.name "Your Name"

export WANDB_API_KEY="YOUR_KEY"
cat .ssh/id_rsa.pub
