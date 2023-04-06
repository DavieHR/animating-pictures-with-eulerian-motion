set -e
if [ $# -ne 1 ]; then
    echo ./watch_exp your_exp_name
fi

exp_name=$1

cat log/${exp_name}/exp_profile.log | xargs -I {} tail -f "log/${exp_name}/"{}".log"

