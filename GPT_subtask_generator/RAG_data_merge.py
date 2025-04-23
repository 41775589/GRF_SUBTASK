import os
import json

# 设置路径
base_dir = "RAG_data/database"
all_data = []

# 遍历所有 rag_task_x 文件夹
for i in range(60):
    folder_name = f"rag_task_{i}"
    file_name = f"RAG_data_{i}.json"
    file_path = os.path.join(base_dir, folder_name, file_name)

    # 读取并拼接数据
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        all_data.extend(data)

# 保存合并后的数据
output_path = "RAG_data/database/merged_data.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print(f"合并完成，共合并了 {len(all_data)} 条数据，保存在 {output_path}")