task="${task:-ovd}" # ovd, fsod, osod
vit="${vit:-l}" # s, b, l
dataset="${dataset:-coco}" # coco, lvis
shot="${shot:-10}"
split="${split:-1}"
num_gpus="${num_gpus:-`nvidia-smi -L | wc -l`}"

echo "task=$task, vit=$vit, dataset=$dataset, shot=$shot, split=$split, num_gpus=$num_gpus"

export WANDB_API_KEY="YOU-KEY-HERE"

# Define the timestamp format (e.g., HH-MM-SS--YYYY-MM-DD)
timestamp=$(date +%Y-%m-%d--%H-%M-%S)

lora=""
# Loop through all arguments
for arg in "$@"
do
    # Check if the argument is --lora
    if [ "$arg" == "--lora" ]; then
        lora="_lora"
        echo "LoRA enabled!"
        break
    fi
done


case $task in

    ovd)
    if [[ "$dataset" == "coco" ]]
    then
        python3 tools/train_net.py    --num-gpus $num_gpus  \
            --config-file configs/open-vocabulary/minicoco/vit${vit}${lora}.yaml \
            MODEL.WEIGHTS  weights/initial/open-vocabulary/vit${vit}+rpn.pth \
            DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
            OUTPUT_DIR output/train/open-vocabulary/minicoco/vit${vit}${lora}/$timestamp
    else
        python3 tools/train_net.py    --num-gpus $num_gpus  --eval-only \
            --config-file  configs/open-vocabulary/lvis/vit${vit}.yaml \
            MODEL.WEIGHTS  weights/initial/open-vocabulary/vit${vit}+rpn_lvis.pth \
            DE.OFFLINE_RPN_CONFIG  configs/RPN/mask_rcnn_R_50_FPN_1x.yaml \
            OUTPUT_DIR output/train/open-vocabulary/lvis/vit${vit}/ $@
    fi
    ;;

    fsod)
        python3 tools/train_net.py --num-gpus $num_gpus  \
            --config-file configs/few-shot/vit${vit}_shot${shot}.yaml  \
            MODEL.WEIGHTS  weights/initial/few-shot/vit${vit}+rpn.pth \
            DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
            OUTPUT_DIR output/train/few-shot/shot-${shot}/vit${vit}/  $@
    ;;

    osod)
        python3 tools/train_net.py \
            --num-gpus $num_gpus \
            --config-file configs/one-shot/split${split}_vit${vit}.yaml \
            MODEL.WEIGHTS weights/initial/oneshot/vit${vit}+rpn.split${split}.pth \
            DE.OFFLINE_RPN_CONFIG  configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
            OUTPUT_DIR  output/train/one-shot/split${split}/vit${vit}/ \
            DE.ONE_SHOT_MODE  True $@
            ;;
    *)
        echo "skip"
        ;;
esac