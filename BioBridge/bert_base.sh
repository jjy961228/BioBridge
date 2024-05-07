for seed in 11111
do
    for model in krbert mbert_cased mbert_uncased xlmr_base kobert 
    do
        for method in base bio bridging 
        do  
            for schedular in linear 
            do
                for warmup_step in 0.1
                do
                    for lr in 6e-5 5e-5 4e-5 3e-5 2e-5 1e-5
                    do
                        for classifier_lr in 2e-2 
                        do
                            for BioEmbLr in 2e-2
                            do
                                CUDA_VISIBLE_DEVICES=0 python main.py --random_seed $seed --model $model --method $method --schedular $schedular --warmup_step $warmup_step --lr $lr --classifier_lr $classifier_lr --BioEmbLr $BioEmbLr
                            done                      
                        done
                    done 
                done
            done
        done
    done
done
