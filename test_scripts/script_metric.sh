echo $1 $2

cd /workspace/

. ds/bin/activate

python dreamsim/metrics.py --prompt_path=$1 --exp_path=$2 --model_path="Mask" 
python dreamsim/metrics.py --prompt_path=$1 --exp_path=$2 --model_path="OneActor"

deactivate ds