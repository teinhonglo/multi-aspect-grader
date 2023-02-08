#!/bin/bash
# dependency: torch, torchaudio, transformers, datasets, librosa
# stage
stage=1
# gpu
gpuid=2

# data config
kfold=5
folds=`seq 1 $kfold`
#scores="content pronunciation vocabulary"
scores="content"
test_book=1
part=1
trans_type=trans_stt_tov_wod

# wav2cec2 config
model_path="facebook/wav2vec2-base"
local_path="exp/icnale/wav2vec2-base/single_label_classification/baseline/holistic/1/best"
# problem type [regression, single_label_classification]
problem_type="regression"
num_labels=1    # when regression set 1, else 8
class_weight_alpha=0.0

# training config
nj=4
train_conf=conf/train_teemi.json
conf=$(basename -s .json $train_conf)

# eval config
bins=""

# visualization config
vi_labels="pre-A,A1,A1A2,A2,A2B1,B1,B1B2,B2"
vi_bins=""

. ./local/parse_options.sh
. ./path.sh

tsv_root=data-speaking/teemi-tb${test_book}p${part}/${trans_type}
json_root=data-json/teemi-tb${test_book}p${part}/${trans_type}
exp_root=exp/teemi-tb${test_book}p${part}/$trans_type/wav2vec2-base/$problem_type/${conf}_new_icnaleft3

if [ "$problem_type" == "regression" ]; then
    bins="1,2,2.5,3,3.5,4,4.5,5"  # no 1.5 score (pre-A-A1)
    vi_bins="2,2.5,3,3.5,4,4.5,5" # below A1(2) is pre-A
fi

if [ $stage -le 0 ]; then
    for score in content pronunciation vocabulary; do
        for fd in `seq 1 5`; do
            for data in train valid test; do
                [ ! -d $json_root/$score/$fd ] && mkdir -p $json_root/$score/$fd
                python local/data_prep_teemi.py \
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
                python train.py  --class-weight-alpha $class_weight_alpha \
                    --train-conf $train_conf \
                    --model-path $model_path \
                    --local-path $local_path \
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
            #[ ! -d $exp_root/$score/$fd/bins9 ] && mkdir -p $exp_root/$score/$fd/bins9
            CUDA_VISIBLE_DEVICES="$gpuid" \
                python test.py --bins "$bins" \
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
    python make_report.py --bins "$bins" --merge-speaker \
        --result_root $exp_root --scores "$scores" --folds "$folds"
fi

if [ $stage -le 4 ]; then
    # produce confusion matrix in $exp_root/score_name.png
    python local/visualization.py \
        --result_root $exp_root --scores "$scores" --folds "$folds" \
       --bins "$vi_bins" --labels "$vi_labels"
    python local/visualization.py --merge-speaker \
        --result_root $exp_root --scores "$scores" --folds "$folds" \
        --bins "$vi_bins" --labels "$vi_labels"
fi
