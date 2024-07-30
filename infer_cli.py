DIR = "/home/skywalker/Cloner/"


import os
from RVC.easy_sync import Channel
logs_folder ='{DIR}drive/MyDrive/project-main/logs'
weights_folder = '{DIR}drive/MyDrive/project-main/assets/weights'
if not "logs_backup" in locals(): logs_backup = Channel('{DIR}RVC/logs',logs_folder,every=30,exclude="mute")
if not "weights_backup" in locals(): weights_backup = Channel('{DIR}RVC/assets/weights',weights_folder,every=30)

# if os.path.exists('{DIR}drive/MyDrive'):
#     if not os.path.exists(logs_folder): os.makedirs(logs_folder)
#     if not os.path.exists(weights_folder): os.makedirs(weights_folder)
#     logs_backup.start()
#     weights_backup.start()


#@title ⛏️ **CREATE TRANING FILES** - This will process the data, extract the features and create your index file for you!

import os
from pytube import YouTube
from IPython.display import clear_output

def calculate_audio_duration(file_path):
    duration_seconds = len(AudioSegment.from_file(file_path)) / 1000.0
    return duration_seconds


os.chdir(f"{DIR}RVC")

#@markdown <small> Give your AI Voice Model a name you won't forget 😉 Avoid weird characters, spaces, symbols, etc.
model_name = 'My-Voice' #@param {type:"string"}
dataset_folder = '{DIR}dataset' #@param {type:"string"}
# or_paste_a_youtube_link=""#@param {type:"string"}
# if or_paste_a_youtube_link !="":
#     youtube_to_wav(or_paste_a_youtube_link)

from pydub import AudioSegment
file_path = dataset_folder
try:
    duration = calculate_audio_duration(file_path)
    if duration < 600:
        cache = True
    else:
        cache = False
except:
    cache = False

while len(os.listdir(dataset_folder)) < 1:
    input("Your dataset folder is empty.")

os.makedirs(f"./logs/{model_name}", exist_ok=True)

with open(f'./logs/{model_name}/preprocess.log','w') as f:
    print("Starting...")

import subprocess

command = f"python infer/modules/train/preprocess.py {dataset_folder} 32000 2 ./logs/{model_name} False 3.0 > /dev/null 2>&1"
result = subprocess.run(command, shell=True, check=True)


with open(f'./logs/{model_name}/preprocess.log','r') as f:
    if 'end preprocess' in f.read():
        clear_output()
        # display(Button(description="\u2714 Success", button_style="success"))
    else:
        print("Error preprocessing data... Make sure your dataset folder is correct.")

f0method = "rmvpe_gpu" # @param ["pm", "harvest", "rmvpe", "rmvpe_gpu"]
addon_path =  f"{DIR}RVC/"
with open(f'{addon_path}logs/{model_name}/extract_f0_feature.log','w') as f:
    print("Starting...")
if f0method != "rmvpe_gpu":
    command = f"python infer/modules/train/extract/extract_f0_print.py ./logs/{model_name} 2 {f0method}" 
    result = subprocess.run(command,shell=True, check=True)
else:
    command = f"python infer/modules/train/extract/extract_f0_rmvpe.py 1 0 0 ./logs/{model_name} True"
    result = subprocess.run(command,shell=True, check=True)

command = f"python infer/modules/train/extract_feature_print.py cuda:0 1 0 ./logs/{model_name} v2 True"
result = subprocess.run(command, shell=True, check=True)


with open(f'{addon_path}logs/{model_name}/extract_f0_feature.log','r') as f:
    if 'all-feature-done' in f.read():
        clear_output()
    else:
        print("Error preprocessing data... Make sure your data was preprocessed.")

import numpy as np
import faiss


os.chdir(f"{DIR}RVC")


def train_index(exp_dir1, version19):
    exp_dir = f"{addon_path}logs/%s" % (exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if not os.path.exists(feature_dir):
        return "请先进行特征提取!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "请先进行特征提取！"
    infos = []
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
        yield "\n".join(infos)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            logger.info(info)
            infos.append(info)
            yield "\n".join(infos)

    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    infos.append("training")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )

    infos.append("adding")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    infos.append(
        "成功构建索引，added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (n_ivf, index_ivf.nprobe, exp_dir1, version19)
    )

training_log = train_index(model_name, 'v2')

for line in training_log:
    print(line)
    if 'adding' in line:
        clear_output()
        # display(Button(description="\u2714 Success", button_style="success"))