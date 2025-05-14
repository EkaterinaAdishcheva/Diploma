echo $1 $2

cd /workspace/

. oa_venv/bin/activate
python Diploma/OneActor/generate_data_target.py --prompt_path=$1 --exp_path=$2
python Diploma/OneActor/generate_data_base.py --prompt_path=$1 --exp_path=$2
python Diploma/OneActor/tune_mask.py --prompt_path=$1 --exp_path=$2 --model_path="Mask" 
python Diploma/OneActor/inference_mask.py --prompt_path=$1 --exp_path=$2 --model_path="Mask"
python Diploma/OneActor/tune_oneactor.py --prompt_path=$1 --exp_path=$2 --model_path="OneActor"
python Diploma/OneActor/inference_oneactor.py --prompt_path=$1 --exp_path=$2 --model_path="OneActor"
deactivate oa_venv

. ds/bin/activate

python dreamsim/metrics.py --prompt_path=$1 --exp_path=$2 --model_path="Mask" 
python dreamsim/metrics.py --prompt_path=$1 --exp_path=$2 --model_path="OneActor"

deactivate ds