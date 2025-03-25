import os

# 定义映射关系
rename_map = {
    0: 0, 1: 10, 2: 11, 3: 12, 4: 13, 5: 14, 6: 15, 7: 16, 8: 17, 9: 18,
    10: 19, 11: 1, 12: 20, 13: 21, 14: 22, 15: 23, 16: 24, 17: 25, 18: 26, 19: 27,
    20: 28, 21: 29, 22: 2, 23: 3, 24: 4, 25: 5, 26: 6, 27: 7, 28: 8, 29: 9
}


def rename_folders(base_path):
    # 先获取所有文件夹名称
    folders = [f for f in os.listdir(base_path) if f.startswith("tb_log_")]

    # 按照旧编号创建完整的路径映射，避免命名冲突
    rename_operations = {}
    for folder in folders:
        old_number = int(folder.split("_")[-1])
        if old_number in rename_map:
            new_folder_name = f"tb_log_{rename_map[old_number]}"
            rename_operations[folder] = new_folder_name

    # 先改成临时名称，防止命名冲突
    temp_operations = {}
    for old_name, new_name in rename_operations.items():
        temp_name = old_name + "_temp"
        os.rename(os.path.join(base_path, old_name), os.path.join(base_path, temp_name))
        temp_operations[temp_name] = new_name

    # 再改回目标名称
    for temp_name, new_name in temp_operations.items():
        os.rename(os.path.join(base_path, temp_name), os.path.join(base_path, new_name))

    print("重命名完成！")


# 运行脚本

base_directory = f"RAG_data/trajectory/rag_task_22"  # 替换为你的文件夹路径
rename_folders(base_directory)
