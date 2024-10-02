import requests
from tqdm import tqdm
import time
import rarfile
import os

__all__ = ['KAIST_clean']
def download_KAIST_clean(rar_save_path='raw_file/KAIST_dataset_clean.rar'):
    """
    Download KAIST dataset cleaned rar file\n
    **Args:**  \n
    - rar_save_path  Default: raw_file/KAIST_dataset_clean.rar\n
    """
    # 下载链接
    url = 'https://ai-studio-online.bj.bcebos.com/v1/17e55c49735048489473905df7e6d2191f177ced9a274700a750bf2c3d54ed19?responseContentDisposition=attachment%3Bfilename%3D%E9%87%8D%E6%96%B0%E6%A0%87%E6%B3%A8%E7%9A%84kaist.rar&authorization=bce-auth-v1%2F5cfe9a5e1454405eb2a975c43eace6ec%2F2024-10-02T04%3A35%3A27Z%2F21600%2F%2Fcd21af41621ba3743877e58e084ca8463f47f85ab9178ad36063c722eadc90af'  # 将此URL替换为实际的下载链接
    response = requests.get(url, stream=True)
    
    # 确保请求成功
    if response.status_code != 200:
        print("Unable to download the file, please check if the URL is correct.")
        return
    
    # 获取文件大小
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 每次读取的数据块大小
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    start_time = time.time()
    downloaded_size = 0

    # 确保保存路径的父目录存在
    save_dir = os.path.dirname(rar_save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(rar_save_path, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
            downloaded_size += len(data)
            
            # 计算并显示下载速度
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:  # 防止除以零
                speed = downloaded_size / elapsed_time / (1024 * 1024)  # 转换为MB/s
                progress_bar.set_postfix_str(f'{speed:.2f} MB/s')
    
    progress_bar.close()
    if total_size != 0 and progress_bar.n != total_size:
        print("Error in downloading")
    print("Download complete")


def extract_rar_file(file_path, output_dir):
    """
    Extract rar file to output_dir\n
    **Args:**  \n
    - file_path*  rar file path\n
    - output_dir*  output directory
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"File {file_path} not exists.")
        return
    
    # 创建输出目录如果它还不存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print(f"{output_dir}Path exists. Overwrite ?")
        answer = input("y/n: ")
        if not answer == 'y':
            return
    
    # 打开RAR文件
    print("Extracting...")
    with rarfile.RarFile(file_path) as rf:
        # 获取所有文件的信息
        members = rf.namelist()
        total_size = sum(member.file_size for member in rf.infolist())
        
        # 使用tqdm创建进度条
        with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            # 解压所有内容到输出目录，并更新进度条
            for member in members:
                rf.extract(member, path=output_dir)
                pbar.update(rf.getinfo(member).file_size)
    
    print(f"File extracted to {output_dir}")

def KAIST_clean(rar_save_path='raw_file/KAIST_dataset_clean.rar', file_save_path='data/KAIST_clean', extract=True):
    """
    Download cleaned KAIST dataset\n
    **If you don't have rar added to PATH, please set extrat = False and extract files manually**\n
    **Args:**  \n
    - rar_save_path  default: raw_file/KAIST_dataset_clean.rar\n
    - file_save_path  default: data/KAIST_clean\n
    - extract  default: True
    """
    # 下载
    if os.path.exists(rar_save_path):
        print(f"{rar_save_path} File exists. Overwrite ?")
        answer = input("y/n: ")
        if answer == 'y':
            download_KAIST_clean(rar_save_path)
    else:
        download_KAIST_clean(rar_save_path)

    # 解压
    if extract:
        extract_rar_file(rar_save_path, file_save_path)
    else:
        print("Set extract FALSE, Please extract files manually")
    print("KAIST dataset prepared")

