if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
# seq_len=96
model_name=PatchTST_MoE_cluster

root_path_name=./dataset/ETT-small/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

random_seed=2021
for seq_len in 96
do
for pred_len in 96
do
for random_seed in 2021
do
for learning_rate in 0.001
do
for T_num_expert in 2
do
for T_top_k in 1
do
for F_num_expert in 2
do
for F_top_k in 1
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --T_num_expert $T_num_expert\
      --T_top_k $T_top_k\
      --F_num_expert $F_num_expert\
      --F_top_k $F_top_k\
      --beta 0.1 \
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --lradj 'TST'\
      --pct_start 0.4 \
      --gpu 5\
      --itr 1 --batch_size 128 --learning_rate $learning_rate >logs/LongForecasting/ettm2/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$T_num_expert'_'$T_top_k'_'$F_num_expert'_'$F_top_k'_'$learning_rate'_0.1'.log
done
done
done
done
done
done
done
done


for seq_len in 96
do
for pred_len in 192
do
for random_seed in 2021
do
for learning_rate in 0.001
do
for T_num_expert in 2
do
for T_top_k in 1
do
for F_num_expert in 2
do
for F_top_k in 1
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --T_num_expert $T_num_expert\
      --T_top_k $T_top_k\
      --F_num_expert $F_num_expert\
      --F_top_k $F_top_k\
      --beta 0.1 \
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --lradj 'TST'\
      --pct_start 0.4 \
      --gpu 5\
      --itr 1 --batch_size 128 --learning_rate $learning_rate >logs/LongForecasting/ettm2/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$T_num_expert'_'$T_top_k'_'$F_num_expert'_'$F_top_k'_'$learning_rate'_0.1'.log
done
done
done
done
done
done
done
done



for seq_len in 96
do
for pred_len in 336
do
for random_seed in 2021
do
for learning_rate in 0.001
do
for T_num_expert in 2
do
for T_top_k in 1
do
for F_num_expert in 2
do
for F_top_k in 1
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --T_num_expert $T_num_expert\
      --T_top_k $T_top_k\
      --F_num_expert $F_num_expert\
      --F_top_k $F_top_k\
      --beta 0.1 \
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --lradj 'TST'\
      --pct_start 0.4 \
      --gpu 5\
      --itr 1 --batch_size 128 --learning_rate $learning_rate >logs/LongForecasting/ettm2/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$T_num_expert'_'$T_top_k'_'$F_num_expert'_'$F_top_k'_'$learning_rate'_0.1'.log
done
done
done
done
done
done
done
done




for seq_len in 96
do
for pred_len in 720
do
for random_seed in 2021
do
for learning_rate in 0.001
do
for T_num_expert in 2
do
for T_top_k in 1
do
for F_num_expert in 2
do
for F_top_k in 1
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --T_num_expert $T_num_expert\
      --T_top_k $T_top_k\
      --F_num_expert $F_num_expert\
      --F_top_k $F_top_k\
      --beta 0.1 \
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --lradj 'TST'\
      --pct_start 0.4 \
      --gpu 5\
      --itr 1 --batch_size 128 --learning_rate $learning_rate >logs/LongForecasting/ettm2/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$T_num_expert'_'$T_top_k'_'$F_num_expert'_'$F_top_k'_'$learning_rate'_0.1'.log
done
done
done
done
done
done
done
done
