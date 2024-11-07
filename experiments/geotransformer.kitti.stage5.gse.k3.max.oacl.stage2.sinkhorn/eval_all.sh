for n in $(seq 70 100); do
        python test.py  --test_epoch=$n
        python eval.py  --test_epoch=$n --method=lgr
        # for i in 250 500 1000 2500 5000; do
        #     python eval.py  --test_epoch=$n --num_corr=$i --method=ransac
        # done
        # python eval.py --test_epoch=$n --num_corr=250 --benchmark=$1 --method=svd
        # python eval.py  --test_epoch=$n --num_corr=5000 --method=ransac
done