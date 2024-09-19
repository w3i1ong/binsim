import os
import shutil
import hashlib
from binsim_dataset.file import FunctionSymbols, extract_symbols as extract_file_symbols, is_elf_file

def simplify_project(project_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    file_md5sum = set()
    for root, dirs, files in os.walk(project_dir):
        for file in files:
            file_path = f"{root}/{file}"
            if os.path.islink(file_path):
                continue
            with open(file_path, 'rb') as f:
                magic = f.read(4)
                if magic != b'\x7fELF':
                    continue
            dst_file = f"{dst_dir}/{os.path.basename(file)}"
            # In some projects, there are multiple files with the same content.
            # We only need to copy one of them.
            md5sum = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
            if md5sum in file_md5sum:
                continue
            file_md5sum.add(md5sum)

            if os.path.exists(dst_file):
                raise FileExistsError(f"File {dst_file} already exists.")
            shutil.copy(file_path, dst_file)

def extract_symbols(project_dir, strip=False):
    for file in os.listdir(project_dir):
        file = os.path.join(project_dir, file)
        if not is_elf_file(file):
            continue
        extract_file_symbols(file, f"{file}.symbols", strip=strip)

