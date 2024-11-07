for n in $(seq 29 29); do
    python test.py  --test_epoch=$n --benchmark=$1  
    python eval.py  --test_epoch=$n --benchmark=$1 --method=lgr
    # for m in 5000; do
    #     python eval.py  --test_epoch=$n --num_corr=$m --benchmark=$1 --method=ransac
    # done
    # python eval.py --test_epoch=$n --num_corr=250 --benchmark=$1 --method=svd
done
