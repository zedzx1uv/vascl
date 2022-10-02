python main.py \
   --resdir ../results/vascl/  \
   --devset sts-b \
   --path_sts_data /mnt/rao/home/zb/SimCSE-main/SentEval/data \
   --datapath /mnt/rao/home/zb/SimCSE-main/data \
   --dataname wiki1m_for_simcse \
   --text1 text \
   --text2 text \
   --bert robertabase \
   --lr 5e-06 \
   --lr_scale 100000 \
   --batch_size 256 \
   --epochs 5 \
   --logging_step 500 \
   --seed 0 \
   --gpuid 0 1 

   # text1=text2, forwarding the same instance twice