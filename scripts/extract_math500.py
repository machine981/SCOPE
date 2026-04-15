import json

def read_jsonl(file_path):
    """读取JSONL文件，返回数据列表"""
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            temp_json = json.loads(line)
            data_list.append(temp_json)
    return data_list

def write_jsonl(data_list, output_path):
    """将数据列表写入JSONL文件"""
    with open(output_path, 'w', encoding='utf-8') as file:
        for item in data_list:
            file.write(json.dumps(item, ensure_ascii=False))
            file.write('\n')
    print(f"文件已生成至 {output_path}，共 {len(data_list)} 条数据。")

def main():
    # 输入文件路径
    input_dir = "./data"
    base_filename = "gsm8k_math500_test_addbox_"
    suffix_range = range(8)  # 0到7

    # 输出文件路径
    output_dir = "./data"
    output_base = "math500_"

    # 存储所有math500数据
    math500_data = []

    # 遍历所有文件
    for suffix in suffix_range:
        input_file = f"{input_dir}/{base_filename}{suffix}.json"
        print(f"正在读取: {input_file}")

        try:
            data_list = read_jsonl(input_file)

            # 提取data_source为"math500"的数据
            for item in data_list:
                if item.get('data_source') == 'math500':
                    math500_data.append(item)

            print(f"  - 从文件中找到 {len([item for item in data_list if item.get('data_source') == 'math500'])} 条math500数据")

        except FileNotFoundError:
            print(f"  - 警告: 文件 {input_file} 不存在，跳过")
        except Exception as e:
            print(f"  - 错误: 读取文件 {input_file} 时出错: {e}")

    # 将数据切片为5个文件
    if math500_data:
        num_files = 5
        data_per_file = len(math500_data) // num_files

        print(f"\n共提取 {len(math500_data)} 条数据，将切片为 {num_files} 个文件，每个文件约 {data_per_file} 条数据\n")

        for i in range(num_files):
            start_idx = i * data_per_file
            end_idx = start_idx + data_per_file if i < num_files - 1 else len(math500_data)
            chunk = math500_data[start_idx:end_idx]
            output_path = f"{output_dir}/{output_base}{i}.json"
            write_jsonl(chunk, output_path)
    else:
        print("警告: 没有找到任何math500数据")

if __name__ == "__main__":
    main()
