set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
echo $exp_name
username=`whoami` 


_timestamp=`date +%Y%m%d%H`
if [ ! -d "log/${exp_name}" ]; then
    
    mkdir -p "log/${exp_name}"
fi

CUDA_VISIBLE_DEVICES=5 python3 train.py \
                --data_root "/data/yinlulu2/glint_pro/dataset/fluid_video/fluid_resize_data_20210801/train_chimney" \
                            "/data/yinlulu2/glint_pro/dataset/fluid_video/fluid_resize_data_20210801/train_valid_waterfall" \
                            "/data/yinlulu2/glint_pro/dataset/fluid_video/fluid_resize_data_20210801/train_mountain" \
                            "/data/yinlulu2/glint_pro/dataset/fluid_video/fluid_resize_data_20210801/train_pollution" \
                            "/data/yinlulu2/glint_pro/dataset/fluid_video/fluid_resize_data_20210801/train_valid_cloud" \
                            "/data/yinlulu2/glint_pro/dataset/fluid_video/fluid_resize_data_20210801/train_river" \
                            "/data/yinlulu2/glint_pro/dataset/fluid_video/fluid_resize_data_20210801/train_smoke" \
                            "/data/yinlulu2/glint_pro/dataset/fluid_video/eulerian_data0816/train" \
                 --mask_dir "/data/yinlulu2/glint_pro/dataset/fluid_video/all_video_masks" \
                 --neg_data_path "/data/yinlulu2/glint_pro/dataset/fluid_video/NegativeSamples"\
                 --neg_prob 0.2 \
                 --snapshots ./snapshots/${exp_name} \
                 --tensorboard ./tensorboard/new_group/${exp_name} \
                 --script_path ./script/train/${exp_name} \
                 --resume_path ./snapshots/gl_0x7_068_resume_new \
                 --learning_rate 3e-5 \
                 --learning_rate_coeffs "Dis,100" \
                 --batchSize 16 \
                 --vis_interval 5 \
                 --data_worker_num 8 \
                 --total_epoch 1000 \
                 --save_interval 1 \
                 --lr_decay_iters 1000 \
                 --flow_file_name "average_flow_pwcpaper.flo" \
                 --no_target_norm \
                 --sep_data \
                > "log/${exp_name}/$_timestamp.log" 2>&1 &

pid=$!
echo $pid
mv "log/${exp_name}/$_timestamp.log" "log/${exp_name}/$pid.log"

echo $pid > "log/${exp_name}/exp_profile.log"


