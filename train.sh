#!/bin/bash
D=300
L=("zh_en" "ja_en" "fr_en" "en_de_15k_V1" "en_fr_15k_V1")

for lang in  ${L[*]}
do
    fast_path="data/$lang/fast.data"
    deep_path="data/$lang/deepwalk.data"
    node2rel_path="data/$lang/node2rel"
    transe_path="data/$lang/"
    out_path="data/$lang/"
    echo "$lang longterm"
    python longterm/main.py --input $deep_path --output ${out_path}longterm.vec --node2rel $node2rel_path --q 0.7 
    echo "$lang transe"
    python train_transe.py --input $transe_path --output $out_path
    echo "$lang fasttext"
    python train_fasttext.py --input $fast_path --output $out_path
done

