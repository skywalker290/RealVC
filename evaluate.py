#@title 🔊 **LISTEN TO YOUR MODEL**
DIR = "/home/skywalker/RealVC/"

import os
DIR = os.getcwd() + "/"

# import os
# import IPython.display as ipd
os.chdir(f"{DIR}RVC")
model_path = f"{DIR}RVC/assets/weights/My-Voice_e125_s8750.pth"#@param {type:"string"}
index_path = f"{DIR}RVC/logs/My-Voice/trained_IVF1596_Flat_nprobe_1_My-Voice_v2.index"#@param {type:"string"}
from colorama import Fore
print(Fore.GREEN + f"{index_path} was found") if os.path.exists(index_path) else print(Fore.RED + f"{index_path} was not found")
pitch = 0 # @param {type:"slider", min:-12, max:12, step:1}
input_path = f"{DIR}RVC/audios/astronauts.mp3"#@param {type:"string"}
if not os.path.exists(input_path):
    raise ValueError(f"{input_path} was not found in your RVC folder.")
os.environ['index_root']  = os.path.dirname(index_path)
index_path = os.path.basename(index_path)
f0_method = "rmvpe" # @param ["rmvpe", "pm", "harvest"]
save_as = f"{DIR}RVC/audios/cli_output.wav"#@param {type:"string"}
model_name = os.path.basename(model_path)
os.environ['weight_root'] = os.path.dirname(model_path)
index_rate = 0.5 # @param {type:"slider", min:0, max:1, step:0.01}
volume_normalization = 0 #param {type:"slider", min:0, max:1, step:0.01}
consonant_protection = 0.5 #param {type:"slider", min:0, max:1, step:0.01}

import subprocess

# command = "mkdir tools/cmd && cp tools/infer_cli.py tools/cmd/"
# result = subprocess.run(command, shell=True, check=True)


command = f"rm -f {save_as}"
result = subprocess.run(command, shell=True, check=True)

# !rm -f $save_as

# command = f"python tools/cmd/infer_cli.py --f0up_key {pitch} --input_path {input_path} --index_path {index_path} --f0method {f0_method} --opt_path {save_as} --model_name {model_name} --index_rate {index_rate} --device "cuda:0" --is_half True --filter_radius 3 --resample_sr 0 --rms_mix_rate $volume_normalization --protect {consonant_protection}"
command = f"python tools/cmd/infer_cli.py --f0up_key {pitch} --input_path {input_path} --index_path {index_path} --f0method {f0_method} --opt_path {save_as} --model_name {model_name} --index_rate {index_rate} --device 'cuda:0' --is_half True --filter_radius 3 --resample_sr 0 --rms_mix_rate {volume_normalization} --protect {consonant_protection}"
result = subprocess.run(command,shell=True, check=True)
# !python tools/cmd/infer_cli.py --f0up_key $pitch --input_path $input_path --index_path $index_path --f0method $f0_method --opt_path $save_as --model_name $model_name --index_rate $index_rate --device "cuda:0" --is_half True --filter_radius 3 --resample_sr 0 --rms_mix_rate $volume_normalization --protect $consonant_protection

show_errors = False #@param {type:"boolean"}
# if not show_errors:
#     ipd.clear_output()
# ipd.Audio(save_as)

print("Output File Generated: ",save_as)