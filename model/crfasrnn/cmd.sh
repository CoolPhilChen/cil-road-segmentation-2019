python pred_cpu.py -d 0 -p ../../output/crf-test/ -e log/snapshot/epoch-0.pth -m test
python pred_cpu.py -d 0 -p ../../output/crf-test/ -e log/snapshot/epoch-last.pth -m test

python pred_cpu.py -d 0 -p ../../output/crf-val/ -e log/snapshot/epoch-last.pth -m val