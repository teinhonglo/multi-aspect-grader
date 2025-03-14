#!/bin/bash
# dependency: torch, torchaudio, transformers, datasets, librosa

# wav2cec2 config
model_path="facebook/wav2vec2-base"
# problem type [regression, single_label_classification]
problem_type="single_label_classification"
num_labels=1
[ "$problem_type" == "single_label_classification" ] && num_labels=8

# data config
kfold=1
folds=`seq 1 $kfold`
#scores="content pronunciation vocabulary"
scores="content"
part=2
tsv_root="data-speaking/gept-p${part}/origin"
json_root="data-json/gept-p${part}/origin"

# training config
nj=4
gpuid=0
train_conf=conf/train_gept.json
exp_root=exp/gept-p2/origin/wav2vec2-base/$problem_type/tanhhead_wd1e-2_weight0.8

# stage
stage=0

. ./local/parse_options.sh
. ./path.sh

if [ $stage -le 0 ]; then
    for score in $scores; do
        for fd in $folds; do
            for data in train valid test; do
                [ ! -d $json_root/$score/$fd ] && mkdir -p $json_root/$score/$fd
                python local/data_prep_gept.py \
                    --part $part \
                    --score $score \
                    --tsv $tsv_root/$fd/$data.tsv \
                    --json $json_root/$score/$fd/$data.json || exit 1
            done
        done
    done
fi

if [ $stage -le 1 ]; then
    for score in $scores; do
        for fd in $folds; do
            CUDA_VISIBLE_DEVICES="$gpuid" \
                python train.py --class-weight-alpha 0.8 \
                    --train-conf $train_conf \
                    --model-path $model_path \
                    --problem-type $problem_type \
                    --num-labels $num_labels \
                    --train-json $json_root/$score/$fd/train.json \
                    --valid-json $json_root/$score/$fd/valid.json \
                    --exp-dir $exp_root/$score/$fd \
                    --nj $nj || exit 1
        done
    done
fi

if [ $stage -le 2 ]; then
    for score in $scores; do
        for fd in $folds; do
            CUDA_VISIBLE_DEVICES="$gpuid" \
                python test.py \
                    --model-path $exp_root/$score/$fd/best \
                    --test-json $json_root/$score/$fd/test.json \
                    --exp-dir $exp_root/$score/$fd \
                    --nj $nj || exit 1
        done
    done
fi
