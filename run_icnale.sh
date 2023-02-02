#!/bin/bash
# dependency: torch, torchaudio, transformers, datasets, librosa

# wav2cec2 config
model_path="facebook/wav2vec2-base"
# problem type [regression, single_label_classification]
problem_type="single_label_classification"
num_labels=1
[ "$problem_type" == "single_label_classification" ] && num_labels=5

# data config
kfold=1
folds=`seq 1 $kfold`
scores="holistic"
tsv_root="data-speaking/icnale/trans_stt_whisperv2_large"
json_root="data-json/icnale/trans_stt_whisperv2_large"

# training config
nj=4
gpuid=0
train_conf=conf/train_icnale.json
exp_root=exp/icnale/wav2vec2-base/$problem_type/baseline
dropout=0.2

# eval config
bins=""

# visualization config
vi_labels="A2,B1_1,B1_2,B2,native"
vi_bins=""

# stage
stage=1

. ./local/parse_options.sh
. ./path.sh

if [ $stage -le 0 ]; then
    for score in $scores; do
        for fd in $folds; do
            for data in train valid test; do
                [ ! -d $json_root/$score/$fd ] && mkdir -p $json_root/$score/$fd
                python local/data_prep_icnale.py \
                    --score $score \
                    --tsv $tsv_root/$fd/$data.tsv \
                    --json $json_root/$score/$fd/$data.json || exit 1
            done
        done
    done
    exit 0
fi

if [ $stage -le 1 ]; then
    for score in $scores; do
        for fd in $folds; do
            CUDA_VISIBLE_DEVICES="$gpuid" \
                python train.py --final-dropout $dropout --resume \
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

if [ $stage -le 3 ]; then
    # produce result in $exp_root/report.log
    python make_report.py --bins "$bins" \
        --result_root $exp_root --scores "$scores" --folds "$folds"
fi

if [ $stage -le 4 ]; then
    # produce confusion matrix in $exp_root/score_name.png
    python local/visualization.py \
        --result_root $exp_root --scores "$scores" --folds "$folds" \
        --bins "$vi_bins" --labels "$vi_labels"
fi
