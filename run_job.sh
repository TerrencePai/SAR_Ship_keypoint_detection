#!/bin/bash
# 开启数据增强要求 mmcv>=2.0.0rc4, <2.1.0.， mmdet>3.0.0rc4

run_det() {
    local pid=$1
    while kill -0 $pid 2>/dev/null; do
        echo "第${ps_num}个进程正在运行..."
        sleep 10
    done
    # 等待进程结束并获取其退出状态
    wait $pid
    exit_status=$?
    # 更新 ps_num
    ps_num=$((ps_num + 1))
}

get_test_images() {
    local from_ann=$1
    local n=$2
    if [[ $from_ann ]]; then
        path_re=${data_root}/$3/*${ann_suffix}
    else
        path_re=${data_root}/$3/*${img_suffix}
    fi
    
    # 使用shuf命令随机选择n张图片
    files=($(shuf -n$n -e ${path_re}))
    image_args=""
    
    for file in "${files[@]}"
    do
        if [[ $from_ann ]]; then
            # 提取文件名（不包括扩展名）
            local filename=$(basename -- "$file")
            filename="${filename%.*}"
            file="${data_root}/$4/${filename}${img_suffix}"
        fi 
        image_args+="${file} "
    done
}

ps_num=1

batch_size=64
data_root='data/rsdd'
descriptor='SuperPoint+Boost-B-attlay3'
img_suffix='.jpg'
expand_piexl=2
nohup python XrayDet/models/keypoint_tranformer.py --print --descriptor ${descriptor} --expand_piexl $expand_piexl --data_root ${data_root}  --weight_decay 1e-4 --print_interval 20 --train_ann_file 'train/'  --test_ann_file 'test/all/' 'test/offshore/' 'test/inshore/'  --num_epochs 100 --img_suffix ${img_suffix} --dataset_class 'ship' --batch_size ${batch_size} > /dev/null 2>&1 &
run_det $!
if [ $exit_status -eq 0 ]; then
    echo "训练完成。执行测试"
    checkpoint="work_dirs/rsdd_${descriptor}_100_bs${batch_size}_best_model_weights_scratch.pth"
    md5sum $checkpoint
    get_test_images 0 10 "test/all/images"
    python XrayDet/models/keypoint_tranformer.py --test_threshold 0.8 --descriptor ${descriptor} --data_root ${data_root} --expand_piexl $expand_piexl --img_suffix ${img_suffix} --checkpoint ${checkpoint} --test_images $image_args
fi

data_root='data/hrsid'
descriptor='SuperPoint+Boost-B-attlay3'
img_suffix='.png'
expand_piexl=2
nohup python XrayDet/models/keypoint_tranformer.py --print --descriptor ${descriptor} --data_root ${data_root} --expand_piexl $expand_piexl --weight_decay 1e-4  --print_interval 20 --train_ann_file 'train/'  'val/'  --test_ann_file 'test/all/' 'test/offshore/' 'test/inshore/'  --num_epochs 100 --img_suffix ${img_suffix} --dataset_class 'ship' --batch_size ${batch_size} > /dev/null 2>&1 &
run_det $!
if [ $exit_status -eq 0 ]; then
    echo "训练完成。执行测试"
    checkpoint="work_dirs/hrsid_${descriptor}_100_bs${batch_size}_best_model_weights_scratch.pth"
    md5sum $checkpoint
    get_test_images 0 10 "test/all/images"
    image_args="${data_root}/test/all/images/18_0_0${img_suffix} ${data_root}/test/all/images/23_0_0${img_suffix} ${data_root}/test/all/images/42_0_0${img_suffix} ${data_root}/test/all/images/41_0_0${img_suffix} ${data_root}/test/all/images/73_0_0${img_suffix} ${data_root}/test/all/images/74_0_0${img_suffix} ${data_root}/test/all/images/94_0_0${img_suffix} ${data_root}/test/all/images/131_0_0${img_suffix} ${data_root}/test/all/images/130_0_0${img_suffix} ${data_root}/test/all/images/134_0_0${img_suffix} ${data_root}/test/all/images/247_0_0${img_suffix} ${data_root}/test/all/images/315_0_0${img_suffix} ${data_root}/test/all/images/372_0_0${img_suffix} ${data_root}/test/all/images/418_0_0${img_suffix} ${data_root}/test/all/images/1254_0_0${img_suffix} ${data_root}/test/all/images/1395_0_0${img_suffix} ${data_root}/test/all/images/1480_0_0${img_suffix} ${data_root}/test/all/images/1630_0_0${img_suffix} ${data_root}/test/all/images/1723_0_0${img_suffix} ${data_root}/test/all/images/1786_0_0${img_suffix} ${data_root}/test/all/images/1817_0_0${img_suffix} ${data_root}/test/all/images/1884_0_0${img_suffix} ${data_root}/test/all/images/41_0_0${img_suffix} ${data_root}/test/all/images/1413_0_0${img_suffix} ${data_root}/test/all/images/1560_0_0${img_suffix} ${data_root}/test/all/images/1649_0_0${img_suffix} ${data_root}/test/all/images/1764_0_0${img_suffix} ${data_root}/test/all/images/1791_0_0${img_suffix} ${data_root}/test/all/images/1826_0_0${img_suffix} ${data_root}/test/all/images/1902_0_0${img_suffix} ${data_root}/test/all/images/499_0_0${img_suffix} ${data_root}/test/all/images/93_0_0${img_suffix}  ${data_root}/test/all/images/1454_0_0${img_suffix} ${data_root}/test/all/images/1626_0_0${img_suffix} ${data_root}/test/all/images/1722_0_0${img_suffix} ${data_root}/test/all/images/1767_0_0${img_suffix} ${data_root}/test/all/images/1814_0_0${img_suffix} ${data_root}/test/all/images/1847_0_0${img_suffix} ${data_root}/test/all/images/1917_0_0${img_suffix} ${data_root}/test/all/images/569_0_0${img_suffix}"
    python XrayDet/models/keypoint_tranformer.py --test_threshold 0.8 --descriptor ${descriptor} --data_root ${data_root} --expand_piexl $expand_piexl --img_suffix ${img_suffix}  --checkpoint ${checkpoint} --test_images $image_args
fi
