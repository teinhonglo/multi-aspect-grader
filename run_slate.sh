#!/bin/bash
# data config
kfold=1
folds=`seq 1 $kfold`
scores="holistic"
tsv_root="data-speaking/slate-p1/trans_stt_whisper_large_v2"
src_json_root="data-json/slate-p1/trans_stt_whipser_large_v2"
json_root="data-json/slate-p1/trans_stt_whisper_large_v2_multi_aspect"
multi_aspect_json_file="/share/nas167/teinhonglo/AcousticModel/spoken_test/asr-esp/data/icnale/icnale_monologue/whisperx_large-v1/aspect_feats.json"

# training config
nj=4
gpuid=1
train_conf=conf/train_slate_baseline_reg_wav2vec2.json
suffix=

# eval bins config
bins="2.25,2.75,3.25,3.75,4.25,4.75,5.25"  # for reg


# visualization bins config
vi_labels="A2,A2+,B1,B1+,B2,B2+,C1,C1+"
vi_bins="2.25,2.75,3.25,3.75,4.25,4.75,5.25"   # for reg

# stage
stage=1
stop_stage=1000

. ./local/parse_options.sh
. ./path.sh

trainset_tag=$(dirname $json_root | xargs basename)
trans_tag=$(basename $json_root)
conf_tag=$(basename -s .json $train_conf)
exp_root=exp/$trainset_tag/$trans_tag/${conf_tag}${suffix}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    for score in $scores; do
        for fd in $folds; do
            src_data_dir=$src_json_root/$score/$fd
            dst_data_dir=$json_root/$score/$fd
            [ ! -d $src_data_dir ] && mkdir -p $src_data_dir
            [ ! -d $dst_data_dir ] && mkdir -p $dst_data_dir

            for data in train valid test dev; do
                python local/data_prep_slate.py \
                    --score $score \
                    --tsv $tsv_root/$fd/$data.tsv \
                    --json $src_data_dir/$data.json || exit 1
                
            done
            
            python local/add_feats_to_json_slate.py \
                --src_data_dir $src_data_dir \
                --dst_data_dir $dst_data_dir \
                --json_files "train.json,valid.json,test.json,dev.json" \
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
                    --bins $bins \
                    --train-conf $train_conf \
                    --train-json $json_root/$score/$fd/train.json \
                    --valid-json $json_root/$score/$fd/dev.json \
                    --exp-dir $exp_root/$score/$fd \
                    --nj $nj || exit 1
        done
    done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    for score in $scores; do
        for fd in $folds; do
            for test_set in dev; do
                CUDA_VISIBLE_DEVICES="$gpuid" \
                    python test.py \
                        --model-path $exp_root/$score/$fd \
                        --test-json $json_root/$score/$fd/${test_set}.json \
                        --exp-dir $exp_root/$score/$fd \
                        --test-set $test_set \
                        --nj $nj || exit 1
            done
        done
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    # produce result in $exp_root/report.log
    for test_set in dev; do
        python make_report.py --bins "$bins" \
            --result_root $exp_root \
            --scores "$scores" --folds "$folds" \
            --test_set $test_set
    done
fi


if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    # produce confusion matrix in $exp_root/score_name.png
    for test_set in dev; do
        python local/visualization.py \
            --result_root $exp_root \
             --scores "$scores" --folds "$folds" \
            --bins "$vi_bins" --labels "$vi_labels" \
            --test_set $test_set
    done
fi

exit 0;

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    python plot_tsne.py \
        --result_root $exp_root/holistic/1
fi
