#!/bin/bash
# dependency: torch, torchaudio, transformers, datasets, librosa

# data config
kfold=1
folds=`seq 1 $kfold`
scores="holistic"
tsv_root="data-speaking/icnale/trans_stt_whisper_large"
src_json_root="data-json/icnale/trans_stt_whisper_large"
json_root="data-json/icnale/trans_stt_whisper_large_multi_aspect"
multi_aspect_json_file="/share/nas167/teinhonglo/AcousticModel/spoken_test/asr-esp/data/icnale/icnale_monologue/whisperx_large-v1/aspect_feats.json"

# training config
nj=4
gpuid=1
train_conf=conf/train_icnale_baseline_cls_wav2vec2.json
suffix=

# eval bins config
bins=""
#bins="1.5,2.5,3.5,4.5"  # for reg

# visualization bins config
vi_labels="A2,B1_1,B1_2,B2,native"
vi_bins=""
#vi_bins="1.5,2.5,3.5,4.5"   # for reg

# stage
stage=1
stop_stage=1000

. ./local/parse_options.sh
. ./path.sh

trans_tag=$(basename $json_root)
conf_tag=$(basename -s .json $train_conf)
exp_root=exp/icnale/$trans_tag/${conf_tag}${suffix}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    for score in $scores; do
        for fd in $folds; do
            src_data_dir=$src_json_root/$score/$fd
            dst_data_dir=$json_root/$score/$fd
            [ ! -d $src_data_dir ] && mkdir -p $src_data_dir
            [ ! -d $dst_data_dir ] && mkdir -p $dst_data_dir

            for data in train valid test; do
                python local/data_prep_icnale.py \
                    --score $score \
                    --tsv $tsv_root/$fd/$data.tsv \
                    --json $src_data_dir/$data.json || exit 1
                
            done
            
            python local/add_feats_to_json_icnale.py \
                --src_data_dir $src_data_dir \
                --dst_data_dir $dst_data_dir \
                --json_files "train.json,valid.json,test.json" \
                --multi_aspect_json_file $multi_aspect_json_file
        done
    done
fi


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    for score in $scores; do
        for fd in $folds; do
            if [ -d $exp_root/$score/$fd/best ]; then
                break
            else
                rm -rf $exp_root/$score/$fd
            fi

            CUDA_VISIBLE_DEVICES="$gpuid" \
                python train.py \
                    --train-conf $train_conf \
                    --bins "$bins" \
                    --train-json $json_root/$score/$fd/train.json \
                    --valid-json $json_root/$score/$fd/valid.json \
                    --exp-dir $exp_root/$score/$fd \
                    --nj $nj || exit 1
        done
    done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    for score in $scores; do
        for fd in $folds; do
            CUDA_VISIBLE_DEVICES="$gpuid" \
                python test.py \
                    --model-path $exp_root/$score/$fd \
                    --test-json $json_root/$score/$fd/test.json \
                    --exp-dir $exp_root/$score/$fd \
                    --nj $nj || exit 1
        done
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    # produce result in $exp_root/report.log
    python make_report.py --bins "$bins" \
        --result_root $exp_root --scores "$scores" --folds "$folds"
fi

exit 0;

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    # produce confusion matrix in $exp_root/score_name.png
    python local/visualization.py \
        --result_root $exp_root --scores "$scores" --folds "$folds" \
        --bins "$vi_bins" --labels "$vi_labels"
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    python plot_tsne.py \
        --result_root $exp_root/holistic/1
fi
