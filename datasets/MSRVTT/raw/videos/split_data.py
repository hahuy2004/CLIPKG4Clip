import os
import shutil

# Thư mục gốc chứa video0.mp4 ... video9999.mp4
src_dir = r"E:\Code KLTN E\CLIP4Clip_Code\Dataset\MSRVTT\MSRVTT\videos\all"

# Thư mục đích
dst_root = r"E:\Code KLTN E\CLIP4Clip_Code\Dataset\MSRVTT\MSRVTT\videos\all_split"

# Số video mỗi folder
chunk_size = 2000

# Tổng số video
total_videos = 10000  # từ video0.mp4 -> video9999.mp4

# Tạo các folder và di chuyển/copy video
for i in range(0, total_videos, chunk_size):
    folder_index = i // chunk_size  # 0..4
    dst_dir = os.path.join(dst_root, f"part_{folder_index+1}")
    os.makedirs(dst_dir, exist_ok=True)

    for j in range(i, min(i+chunk_size, total_videos)):
        filename = f"video{j}.mp4"
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)  # dùng move
            # Nếu muốn copy thay vì move thì dùng: shutil.copy2(src_path, dst_path)

print("Done splitting into 5 folders.")
