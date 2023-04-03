set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
echo $exp_name
username=`whoami` 

_timestamp=`date +%Y%m%d%H`
if [ ! -d "log/${exp_name}" ]; then
    mkdir -p "log/${exp_name}"
fi

if [ ! -d "results/${exp_name}" ]; then
    mkdir -p "results/${exp_name}"
fi

CUDA_VISIBLE_DEVICES=0 python3 test.py \
                 --input /data/yinlulu2/glint_pro/dataset/fluid_video/test_img/testing_images/24.png \
                 --EMotion_name "MotionEstimationNet" \
                 --resume_path ./snapshots/gl_0x7_071\
                 --separation_emotion_path "/data/wanghaoran2/PictureAnimination/motion_estimation/checkpoints/20210824_global-whr-no_norm-feat_match_no_mse_large_gan_low_lr/pix2pixHD_894_epoch_net_G.pth" \
                 --output ./results/${exp_name} \
                 --save_file_name "071_old_normal_512_894_0.5.mp4" \
                 --verbose \
                 --frames 60 \
                 --intensity 0.5 \
                 --resolution 512 512 \
                > "log/${exp_name}/$_timestamp.log" 2>&1 &

pid=$!
echo $pid
mv "log/${exp_name}/$_timestamp.log" "log/${exp_name}/$pid.log"
echo $pid > "log/${exp_name}/exp_profile.log"


