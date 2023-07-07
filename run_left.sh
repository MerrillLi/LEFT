# Seattle
python run_left.py --model LTP --dataset seattle --num_users 99 --num_items 99 --num_times 688 --density 0.05 --rank 50 --window 16
python run_left.py --model LTP --dataset seattle --num_users 99 --num_items 99 --num_times 688 --density 0.10 --rank 50 --window 16
python run_left.py --model LTP --dataset seattle --num_users 99 --num_items 99 --num_times 688 --density 0.20 --rank 50 --window 16

# WS-DREAM
python run_left.py --model LTP --dataset wsdream --num_users 142 --num_items 4500 --num_times 64 --density 0.05 --rank 30 --window 8
python run_left.py --model LTP --dataset wsdream --num_users 142 --num_items 4500 --num_times 64 --density 0.10 --rank 30 --window 8
python run_left.py --model LTP --dataset wsdream --num_users 142 --num_items 4500 --num_times 64 --density 0.20 --rank 30 --window 8

# Abilene
python run_left.py --model LTP --dataset abilene --num_users 144 --num_items 168 --num_times 288 --density 0.05 --rank 50 --window 12
python run_left.py --model LTP --dataset abilene --num_users 144 --num_items 168 --num_times 288 --density 0.10 --rank 50 --window 12
python run_left.py --model LTP --dataset abilene --num_users 144 --num_items 168 --num_times 288 --density 0.20 --rank 50 --window 12
