import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("./model/tokenizer")

def check_dataset_token_length(jsonl_path, tokenizer, max_length=512):
    token_lengths = []
    over_length_count = 0
    total_count = 0
    
    # 读取jsonl文件
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="处理数据"):
            try:
                data = json.loads(line)
                total_count += 1
                
                # 获取文本
                text = data["text"]
                
                # 计算token长度
                tokens = tokenizer.encode(text)
                length = len(tokens)
                token_lengths.append(length)
                
                if length > max_length:
                    over_length_count += 1
                    print(f"发现超长数据，长度为: {length}")
                    
            except json.JSONDecodeError:
                print(f"跳过无效的JSON行")
            except KeyError:
                print(f"数据中没有'text'字段")
    
    # 结果统计
    print(f"\n总数据条数: {total_count}")
    print(f"超过{max_length} tokens的数据条数: {over_length_count}")
    print(f"超长数据比例: {over_length_count/total_count*100:.2f}%")
    
    if token_lengths:
        print(f"最大token长度: {max(token_lengths)}")
        print(f"最小token长度: {min(token_lengths)}")
        print(f"平均token长度: {sum(token_lengths)/len(token_lengths):.2f}")
    
    # 绘制长度分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths, bins=50)
    plt.axvline(x=max_length, color='r', linestyle='--', label=f'max_length={max_length}')
    plt.xlabel('Token Length')
    plt.ylabel('Count')
    plt.title('Distribution of Token Lengths')
    plt.legend()
    
    plt.savefig("token_length_distribution.png")
    print("图像已保存为 token_length_distribution.png")
    
    return token_lengths, over_length_count, total_count


token_lengths, over_count, total = check_dataset_token_length("./data/processed_c4_en_1m.jsonl", tokenizer)