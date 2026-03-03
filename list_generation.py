import os

# ======== 配置区域 ========
folder1 = "dataset/with_film/data_film_lighted"   # 第一个文件夹路径 data
folder2 = "dataset/with_film/data_templates_gc"   # 第二个文件夹路径 template
folder3 = "dataset/with_film/annotations_gc"   # 第三个文件夹路径 annotations
output_txt = "dataset/test_list.txt"   # 输出 txt 文件路径
# =========================

def get_relative_files(folder):
    """获取文件夹内所有文件的【相对路径】列表（相对当前脚本路径）"""
    base_path = os.path.abspath(os.path.dirname(__file__))
    folder_path = os.path.join(base_path, folder)
    file_list = []

    for root, dirs, files in os.walk(folder_path):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, base_path)
            file_list.append(rel_path)

    return sorted(file_list)


def main():
    files1 = get_relative_files(folder1)
    files2 = get_relative_files(folder2)
    files3 = get_relative_files(folder3)

    # 检查三个文件夹文件数量是否一致
    max_len = max(len(files1), len(files2), len(files3))

    # 对齐文件列表长度（不足的用空字符串填充）
    files1 += [""] * (max_len - len(files1))
    files2 += [""] * (max_len - len(files2))
    files3 += [""] * (max_len - len(files3))

    with open(output_txt, "w", encoding="utf-8") as f:
        for a, b, c in zip(files1, files2, files3):
            line = a
            if b:
                line += " " + b
            if c:
                line += " " + c
            f.write(line + "\n")

    print(f"写入完成：{output_txt}")


if __name__ == "__main__":
    main()
