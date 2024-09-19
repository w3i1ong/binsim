import os
import glob
from multiprocessing import Pool
from binsim_dataset.project import simplify_project, extract_symbols as extract_symbols_project

def simplify_single_project_dir(args):
    project_dir, src_dir, dst_dir, strip = args
    rel_path = project_dir[len(src_dir) + 1:]
    project, version, arch, platform, compiler, opt_level = rel_path.split('/')
    rel_path = f"{project}/{version}/{arch}/{platform}/{compiler}/{opt_level}"
    dst_path = f"{dst_dir}/{rel_path}"
    simplify_project(project_dir, dst_path)
    extract_symbols_project(dst_path, strip=strip)


def simplify_dataset(src_dir, dst_dir, project_name=None, strip=False, num_workers=20):
    src_dir = os.path.abspath(src_dir)
    if project_name is None:
        paths = glob.glob(f"{src_dir}/*/*/*/*/*/*")
    else:
        paths = glob.glob(f"{src_dir}/{project_name}/*/*/*/*/*")

    with Pool(num_workers) as pool:
        pool.map(simplify_single_project_dir, [(path, src_dir, dst_dir, strip) for path in paths])


def extract_symbols(dataset_dir, project_name=None, num_workers=20):
    dataset_dir = os.path.abspath(dataset_dir)
    if project_name is None:
        paths = glob.glob(f"{dataset_dir}/*/*/*/*/*/*")
    else:
        paths = glob.glob(f"{dataset_dir}/{project_name}/*/*/*/*/*")
    with Pool(num_workers) as pool:
        pool.map(extract_symbols_project, paths)


