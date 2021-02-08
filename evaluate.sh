#!/bin/bash
L=("zh_en" "ja_en" "fr_en" "en_fr_15k_V1" "en_de_15k_V1")

for lang in  ${L[*]}
do
    input_path="data/$lang/"
    out_path="result/final.txt"
    echo "$lang evaluate"
    python evaluate.py --input $input_path --output $out_path
done