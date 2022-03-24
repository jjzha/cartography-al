<<<<<<< HEAD
EXP_PATH=project/resources/logs/trec

SEEDS=( 398048 127003 259479 869323 570852 )
FUNCTIONS=( "random" "entropy" "leastconfidence" "bald" "discriminative" "cartography" )
#FUNCTIONS=( "discriminative" )
=======
EXP_PATH=project/resources/logs/trec_check_instances

SEEDS=( 398048 127003 259479 869323 570852 )
FUNCTIONS=( "random" "entropy" "leastconfidence" "bald" "discriminative" "cartography" )

>>>>>>> ebd1d642135f9e01946c117f65afb47e7810230e
# iterate over seeds
for rsd_idx in "${!SEEDS[@]}"; do
  # iterate over encoders
  for enc_idx in "${!FUNCTIONS[@]}"; do
    echo "Experiment: '${FUNCTIONS[$enc_idx]}' and random seed ${SEEDS[$rsd_idx]}."
    echo "Training MLP classifer using '${FUNCTIONS[$enc_idx]}' acquisition function and random seed ${SEEDS[$rsd_idx]}."

    exp_dir=$EXP_PATH/function-${FUNCTIONS[$enc_idx]}-rs${SEEDS[$rsd_idx]}
    python3 main.py --task trec \
                    --initial_size 500 \
                    --batch_size 16 \
                    --learning_rate_main 0.0001 \
                    --learning_rate_binary 0.00005 \
                    --epochs 30 \
<<<<<<< HEAD
                    --al_iterations 30 \
=======
                    --al_iterations 20 \
>>>>>>> ebd1d642135f9e01946c117f65afb47e7810230e
                    --seed ${SEEDS[$rsd_idx]} \
                    --pretrained \
                    --freeze \
                    --acquisition ${FUNCTIONS[$enc_idx]} \
                    --exp_path ${exp_dir} \
                    --analysis

  done
done

#python3 main.py --task trec --initial_size 500 --plot_results
