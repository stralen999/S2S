if [ $# -eq 0 ]
  then
    printf "Please specify the name of the environment to create.\n	$ bash config.sh <env_name>\n"
    exit
fi

#wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
#bash Anaconda3-2023.09-0-Linux-x86_64.sh -b -p $HOME/anaconda3
#rm -f Anaconda3-2023.09-0-Linux-x86_64.sh

source $HOME/anaconda3/bin/activate
conda create -n $1 python=3.10
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate $1

#conda packages
conda install -c conda-forge ocl-icd-syste==m1.0.0
#pip packages
pip install -r requirements.txt

mkdir $HOME/anaconda3/envs/$1/etc/conda/
mkdir $HOME/anaconda3/envs/$1/etc/conda/activate.d

printf "export PATH_DATA=$HOME/S2S/data/ \n export PATH_MODELS=$HOME/S2S/saved_models/ \n export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$HOME/anaconda3/envs/$1/lib/"> $HOME/anaconda3/envs/$1/etc/conda/activate.d/export.sh
