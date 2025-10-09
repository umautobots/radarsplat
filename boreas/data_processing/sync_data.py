import os
import argparse
import subprocess

def rsync_folder(src, dst, name):
    if not os.path.isdir(src):
        print(f"[WARNING] Folder '{name}' not found at {src}, skipping...")
        return
    print(f"[INFO] Syncing folder: {name}")
    subprocess.run(["rsync", "-avh", "--delete", src + "/", dst + "/"], check=True)


def rsync_png_only(src_folder, dst_folder):
    subprocess.run([
        "rsync", "-avh",
        "--include=*.png",
        "--exclude=*",
        f"{src_folder}/", f"{dst_folder}/"
    ], check=True)

def rsync_flat_png_files(src_folder, dst_folder):
    if not os.path.isdir(src_folder):
        print(f"[WARNING] radar folder not found at {src_folder}, skipping radar images...")
        return

    os.makedirs(dst_folder, exist_ok=True)
    png_files = [f for f in os.listdir(src_folder) if f.endswith(".png") and os.path.isfile(os.path.join(src_folder, f))]

    if not png_files:
        print("[WARNING] No .png files found directly under radar/.")
        return

    print(f"[INFO] Copying {len(png_files)} .png files from radar/ to images/")
    for fname in png_files:
        src_file = os.path.join(src_folder, fname)
        dst_file = os.path.join(dst_folder, fname)
        subprocess.run(["rsync", "-avh", src_file, dst_file], check=True)

def main(path1, path2):
    # 1. Copy radar/*.png -> images/
    radar_src = os.path.join(path1, "radar")
    images_dst = os.path.join(path2, "images")
    print("[INFO] Copying radar to images")
    # rsync_flat_png_files(radar_src, images_dst) # slow
    rsync_png_only(radar_src, images_dst)

    # 2. Sync folders
    folders = [
        "synced_lidar",
        "synced_lidar_map_win5",
        "multipath_model",
        "radar_average_map",
        "baseline_polar_occ_filtered_polar"
    ]
    for folder in folders:
        src = os.path.join(path1, folder)
        dst = os.path.join(path2, folder)
        rsync_folder(src, dst, folder)

    # 3. Copy radar_trajectory.tum
    traj_src = os.path.join(path1, "radar_trajectory.tum")
    traj_dst = os.path.join(path2, "radar_trajectory.tum")
    if os.path.exists(traj_src):
        print("[INFO] Copying radar_trajectory.tum")
        subprocess.run(["rsync", "-avh", traj_src, traj_dst], check=True)
    else:
        print("[WARNING] File 'radar_trajectory.tum' not found, skipping...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path1", help="Source directory (PATH1)")
    parser.add_argument("path2", help="Destination directory (PATH2)")
    args = parser.parse_args()

    main(args.path1, args.path2)
