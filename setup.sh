#!/bin/bash

# Ensure pip is installed at the correct version
pip install pip==22.0

# Set the directory path
DIR=$(pwd)
DIR="${DIR}/"
# DIR="/home/darth/RealVC/"

mkdir ${DIR}

# Variables
var="WebUI"
test="Voice"
c_word="Conversion"
r_word="Retrieval"

# Clone the repository
git clone https://github.com/RVC-Project/${r_word}-based-${test}-${c_word}-${var} ${DIR}RVC

# Install aria2 if not already installed
sudo apt -y install -qq aria2

mkdir ${DIR}RVC/tools/cmd 
cp ${DIR}RVC/tools/infer_cli.py ${DIR}RVC/tools/cmd/infer_cli.py

# Define pretrain files
pretrains=("f0D32k.pth" "f0G32k.pth")
new_pretrains=("f0Ov2Super32kD.pth" "f0Ov2Super32kG.pth")

# Download pretrain files if they do not exist
for file in "${pretrains[@]}"; do
    if [ ! -f "${DIR}RVC/assets/pretrained_v2/${file}" ]; then
        command="aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/${file} -d ${DIR}RVC/assets/pretrained_v2 -o ${file}"
        eval $command
    fi
done

for file in "${new_pretrains[@]}"; do
    if [ ! -f "${DIR}RVC/assets/pretrained_v2/${file}" ]; then
        command="aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/poiqazwsx/Ov2Super32kfix/resolve/main/${file} -d ${DIR}RVC/assets/pretrained_v2 -o ${file}"
        eval $command
    fi
done

# Create directories if they do not exist
mkdir -p ${DIR}dataset
mkdir -p ${DIR}RVC/audios

# Download files
wget -nc https://raw.githubusercontent.com/RejektsAI/EasyTools/main/original -O ${DIR}RVC/original.py
wget -nc https://raw.githubusercontent.com/RejektsAI/EasyTools/main/app.py -O ${DIR}RVC/demo.py
wget -nc https://raw.githubusercontent.com/RejektsAI/EasyTools/main/easyfuncs.py -O ${DIR}RVC/easyfuncs.py
wget -nc https://huggingface.co/Rejekts/project/resolve/main/download_files.py -O ${DIR}RVC/download_files.py
wget -nc https://huggingface.co/Rejekts/project/resolve/main/a.png -O ${DIR}RVC/a.png
wget -nc https://huggingface.co/Rejekts/project/resolve/main/easy_sync.py -O ${DIR}RVC/easy_sync.py
wget -nc https://huggingface.co/spaces/Rejekts/RVC_PlayGround/raw/main/app.py -O ${DIR}RVC/playground.py
wget -nc https://huggingface.co/spaces/Rejekts/RVC_PlayGround/raw/main/tools/useftools.py -O ${DIR}RVC/tools/useftools.py
wget -nc https://huggingface.co/Rejekts/project/resolve/main/astronauts.mp3 -O ${DIR}RVC/audios/astronauts.mp3
wget -nc https://huggingface.co/Rejekts/project/resolve/main/somegirl.mp3 -O ${DIR}RVC/audios/somegirl.mp3
wget -nc https://huggingface.co/Rejekts/project/resolve/main/someguy.mp3 -O ${DIR}RVC/audios/someguy.mp3
wget -nc https://huggingface.co/Rejekts/project/resolve/main/unchico.mp3 -O ${DIR}RVC/audios/unchico.mp3
wget -nc https://huggingface.co/Rejekts/project/resolve/main/unachica.mp3 -O ${DIR}RVC/audios/unachica.mp3

# Run the download_files.py script
cd ${DIR}RVC && python ${DIR}RVC/download_files.py

# Install requirements if not already installed
if [ -z "$installed" ]; then
    cd ${DIR}RVC && pip install -r requirements.txt
    pip install mega.py gdown==4.6.0 pytube pydub gradio==3.42.0
    installed=True
fi
