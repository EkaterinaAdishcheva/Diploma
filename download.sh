result=${PWD##*/}
Y="_download"

python get_mask.py

mkdir "$result$Y"
cp target_mask.pkl "$result$Y"
cp target.jpg "$result$Y"
mkdir "$result$Y/Mask"
mkdir "$result$Y/OneActor"
mkdir "$result$Y/Mask/inference"
mkdir "$result$Y/OneActor/inference"

cp Mask/metrics.csv "$result$Y/Mask"
cp OneActor/metrics.csv "$result$Y/OneActor"
cp Mask/inference/* "$result$Y/Mask/inference"
cp OneActor/inference/* "$result$Y/OneActor/inference"


