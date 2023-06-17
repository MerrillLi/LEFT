# Harvard226
python run_math.py --model CPsgd --dataset harvard226 --num_users 226 --num_items 226 --num_times 864 --density 0.1 --rank 20 --lr 1e-4 --epochs 15000
# GEANT
python run_math.py --model CPsgd --dataset geant_rs --num_users 529 --num_items 112 --num_times 96 --density 0.1 --rank 40 --lr 5e-4 --epochs 15000
# Seattle
python run_math.py --model CPsgd --dataset seattle --num_users 99 --num_items 99 --num_times 688 --density 0.1 --rank 10 --lr 3e-4 --epochs 15000
# WS-DREAM
python run_math.py --model CPsgd --dataset wsdream --num_users 142 --num_items 4500 --num_times 64 --density 0.1 --rank 20 --lr 6e-5 --epochs 15000
# Abilene
python run_math.py --model CPsgd --dataset abilene_rs --num_users 144 --num_items 168 --num_times 288 --density 0.1 --rank 30 --lr 2e-4 --epochs 15000

