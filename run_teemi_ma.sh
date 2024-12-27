#!/bin/bash
# dependency: torch, torchaudio, transformers, datasets, librosa
# stage
stage=1

# data config
kfold=5
folds=`seq 5 $kfold`
scores="content pronunciation vocabulary"
test_book=1
part=1
trans_type=trans_stt   # trans_stt_tov -> cls, trans_stt_tov_wod -> reg
multi_aspect_json_file="/share/nas167/teinhonglo/AcousticModel/spoken_test/asr-esp/data/teemi/teemi_a01/whisperx_large-v1/aspect_feats.json"

# training config
nj=4
gpuid=0
train_conf=conf/train_teemi_baseline_cls_wav2vec2.json
suffix=

# eval config
bins=""    # for cls
#bins="1,2,2.5,3,3.5,4,4.5,5"  # for reg, no 1.5 score (pre-A-A1)

# visualization config
vi_bins="" # for cls
#vi_bins="2,2.5,3,3.5,4,4.5,5" # for reg, below A1(2) is pre-A
vi_labels="pre-A,A1,A1A2,A2,A2B1,B1,B1B2,B2"

. ./local/parse_options.sh
. ./path.sh

tsv_root=data-speaking/teemi-tb${test_book}p${part}/${trans_type}
src_json_root=data-json/teemi-tb${test_book}p${part}/${trans_type}
json_root=data-json/teemi-tb${test_book}p${part}/${trans_type}_multi_aspect

conf_tag=$(basename -s .json $train_conf)
exp_root=exp/teemi-tb${test_book}p${part}/$trans_type/${conf_tag}${suffix}

if [ $stage -le 0 ]; then
    for score in content pronunciation vocabulary; do
        for fd in `seq 1 5`; do
            src_data_dir=$src_json_root/$score/$fd
            dst_data_dir=$json_root/$score/$fd
            [ ! -d $src_data_dir ] && mkdir -p $src_data_dir
            [ ! -d $dst_data_dir ] && mkdir -p $dst_data_dir

            for data in train valid test; do
                python local/data_prep_teemi.py \
                    --score $score \
                    --tsv $tsv_root/$fd/$data.tsv \
                    --json $src_data_dir/$data.json || exit 1
            
            done
            
            python local/add_feats_to_json_teemi.py \
                --src_data_dir $src_data_dir \
                --dst_data_dir $dst_data_dir \
                --json_files "train.json,valid.json,test.json" \
                --multi_aspect_json_file $multi_aspect_json_file
        done
    done
fi

if [ $stage -le 1 ]; then
    for score in $scores; do
        for fd in $folds; do
            CUDA_VISIBLE_DEVICES="$gpuid" \
                python train.py  \
                    --train-conf $train_conf \
                    --bins "$bins" \
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
            #--train-conf $exp_root/$score/$fd/train_conf.json \
            CUDA_VISIBLE_DEVICES="$gpuid" \
                python test.py \
                    --model-path $exp_root/$score/$fd \
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
