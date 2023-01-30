#!/bin/bash
# dependency: torch, torchaudio, transformers, datasets, librosa

# data config
kfold=5
folds=`seq 1 $kfold`
scores="content pronunciation vocabulary"
#scores="pronunciation vocabulary"
test_book=1
part=2
trans_type=trans_stt_tov_wod
tsv_root=data-speaking/teemi-tb${test_book}p${part}/${trans_type}
json_root=data-json/teemi-tb${test_book}p${part}/${trans_type}

# wav2cec2 config
model_path="facebook/wav2vec2-base"
# problem type [regression, single_label_classification]
problem_type="regression"
num_labels=1
[ "$problem_type" == "single_label_classification" ] && num_labels=9

# training config
nj=4
gpuid=0
train_conf=conf/train_teemi.json
conf=$(basename -s .json $train_conf)
exp_root=exp/teemi-tb${test_book}p${part}/$trans_type/wav2vec2-base/$problem_type/${conf}

# eval config
bins="1,2,2.5,3,3.5,4,4.5,5" # no 1.5 score (pre-A-A1)
#bins="1,2,3,4,5"

# visualization config
vi_bins="2,2.5,3,3.5,4,4.5,5" # below A1(2) is pre-A
vi_labels="pre-A,A1,A1A2,A2,A2B1,B1,B1B2,B2"

# stage
stage=0

. ./local/parse_options.sh
. ./path.sh

if [ $stage -le 0 ]; then
    for score in $scores; do
        for fd in $folds; do
            for data in train valid test; do
                [ ! -d $json_root/$score/$fd ] && mkdir -p $json_root/$score/$fd
                python local/data_prep_teemi.py \
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
                python train.py  \
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
            #[ ! -d $exp_root/$score/$fd/bins9 ] && mkdir -p $exp_root/$score/$fd/bins9
            CUDA_VISIBLE_DEVICES="$gpuid" \
                python test.py --bins $bins \
                    --model-path $exp_root/$score/$fd/best \
                    --test-json $json_root/$score/$fd/test.json \
                    --exp-dir $exp_root/$score/$fd \
                    --nj $nj || exit 1
        done
    done
fi

if [ $stage -le 3 ]; then
    # produce result in $exp_root/report.log
    python local/make_report.py \
        --result_root $exp_root --scores "$scores" --folds "$folds"
fi
exit 0
if [ $stage -le 4 ]; then
    # produce confusion matrix in $exp_root/score_name.png
    python local/visualization.py \
        --result_root $exp_root --scores "$scores" --folds "$folds" \
        --bins "$vi_bins" --labels "$vi_labels"
fi
