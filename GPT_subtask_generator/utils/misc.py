import subprocess
import os
import json
import logging

from utils.extract_task_code import file_to_string

def set_freest_gpu():
    freest_gpu = get_freest_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(freest_gpu)

def get_freest_gpu():
    sp = subprocess.Popen(['gpustat', '--json'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str, _ = sp.communicate()
    gpustats = json.loads(out_str.decode('utf-8'))
    # Find GPU with most free memory
    freest_gpu = min(gpustats['gpus'], key=lambda x: x['memory.used'])

    return freest_gpu['index']

def filter_traceback(s):
    # lines = s.split('\n')
    # filtered_lines = []
    # for i, line in enumerate(lines):
    #     if line.startswith('Traceback'):
    #         for j in range(i, len(lines)):
    #             # if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
    #             #     break
    #             filtered_lines.append(lines[j])
    #         return '\n'.join(filtered_lines)
    # return ''  # Return an empty string if no Traceback is found
    lines = s.split('\n')
    last_traceback_index = -1  # 记录最后一个 'Traceback' 位置

    # 反向查找最后一个 'Traceback'
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith('Traceback'):
            last_traceback_index = i
            break  # 找到最后一个 'Traceback'，立即退出循环

    # 如果找到了 Traceback，就提取它及其后续内容
    if last_traceback_index != -1:
        return '\n'.join(lines[last_traceback_index:])

    return ''  # 没有找到 Traceback，返回空字符串

def block_until_training(rl_filepath, log_status=False, iter_num=-1, response_id=-1):
    # Ensure that the RL training has started before moving on
    while True:
        rl_log = file_to_string(rl_filepath)
        if "FPS:" in rl_log or "Traceback" in rl_log:
            if log_status and "FPS:" in rl_log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} successfully training!")
            if log_status and "Traceback" in rl_log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            break

if __name__ == "__main__":
    print(get_freest_gpu())