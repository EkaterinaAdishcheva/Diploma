python Diploma/OneActor/generate_data_target.py --prompt_path=Diploma/config/prompt-adventurer.yaml --exp_path="adventurer_1"

python Diploma/OneActor/generate_data_base.py --prompt_path=Diploma/config/prompt-adventurer.yaml --exp_path="adventurer_1"

python Diploma/OneActor/tune_mask.py --prompt_path=Diploma/config/prompt-adventurer.yaml --exp_path="adventurer_1" --model_path="Mask" 

python Diploma/OneActor/inference_mask.py --prompt_path=Diploma/config/prompt-adventurer.yaml --exp_path="adventurer_1" --model_path="Mask"
 
python Diploma/OneActor/tune_oneactor.py --prompt_path=Diploma/config/prompt-adventurer.yaml --exp_path="adventurer_1" --model_path="OneActor"

python Diploma/OneActor/inference_oneactor.py --prompt_path=Diploma/config/prompt-adventurer.yaml --exp_path="adventurer_1" --model_path="OneActor"
