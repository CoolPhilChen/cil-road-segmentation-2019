python pred_cpu.py -d 0 -p ../../output/psp-test/ -e log/snapshot/epoch-75.pth -m test
python pred_cpu.py -d 0 -p ../../output/psp-test/ -e log/snapshot/epoch-last.pth -m test

python pred_cpu.py -d 0 -p ../../output/psp-val/ -e log/snapshot/epoch-last.pth -m val