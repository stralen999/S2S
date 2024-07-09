# stconvs2s   

1.source ../INTAKE-Baselines/venv/bin/activate


train:
2.python main.py -fh 1 -e 1 -b 2 -fd 


forecast:
3.python main.py -fh 1 -e 1 -b 2 -fd 07-Fev-2021

in point 3. the date in fd is the T+7 ; corresponds therefore to T=31-Jan-2021


