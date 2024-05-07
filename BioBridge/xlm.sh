for seed in 11111
do
    for model in xlmr_large xlm 
    do
        for method in bribio base
        do  
            for schedular in linear 
            do
                for warmup_step in 0.1
                do
                    for lr in 2e-6 3e-6 5e-6 1e-5 5e-5
                    do
                        for classifier_lr in 2e-2 
                        do
                            for BioEmbLr in 2e-2
                            do
                                CUDA_VISIBLE_DEVICES=1 python main.py --run_wandb True --random_seed $seed --model $model --method $method --schedular $schedular --warmup_step $warmup_step --lr $lr --classifier_lr $classifier_lr --BioEmbLr $BioEmbLr
                            done                      
                        done
                    done 
                done
            done
        done
    done
done
