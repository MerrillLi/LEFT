import os

def delete_files_in_folder(folder_path):
    """递归删除指定文件夹下的所有文件"""

    # 获取文件夹下的所有文件和子文件夹
    files = os.listdir(folder_path)

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)

        # 判断是否是文件
        if os.path.isfile(file_path):
            os.remove(file_path)

        # 如果是目录则递归调用自己
        elif os.path.isdir(file_path):
            delete_files_in_folder(file_path)


if __name__ == '__main__':
    delete_files_in_folder('./results')
