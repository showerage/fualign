#!/bin/bash
L=("zh_en" "ja_en" "fr_en" "en_de_15k_V1" "en_fr_15k_V1")
for lang in  ${L[*]}
do
    echo "process $lang"
    python preproccess.py --l $lang
done