  # CUDA_VISIBLE_DEVICES=1
python main.py \
   --resdir ../results/vascl/  \
   --devset sts-b \
   --path_sts_data /media/rao/Disk-1/home/zb/SimCSE/SentEval/data \
   --datapath /media/rao/Disk-1/home/zb/SimCSE/data \
   --dataname wiki1m_for_simcse \
   --text1 text \
   --text2 text \
   --bert bertbase \
   --lr 5e-06 \
   --lr_scale 100000 \
   --batch_size 128 \
   --epochs 5 \
   --logging_step 1 \
   --seed 3407 \
   --gpuid 0 1 \
   --advk 30 \
   --freelb 


   # text1=text2, forwarding the same instance twice