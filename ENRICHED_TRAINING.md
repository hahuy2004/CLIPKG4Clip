# Hướng Dẫn Sử Dụng Enriched Training

## Tổng Quan

Code đã được cập nhật để hỗ trợ training hai giai đoạn:
1. **STAGE 1**: Training trên enriched data (dữ liệu đã được làm giàu bằng caption sinh tự động)
2. **STAGE 2**: Fine-tuning trên original data (dữ liệu gốc) sử dụng checkpoint tốt nhất từ STAGE 1

## Cách Sử Dụng

### 1. MSVD Dataset

**Yêu cầu:**
- File enriched captions: `enriched-caption-complete.pkl` (đặt trong thư mục data_path)
- File raw captions: `raw-captions.pkl` (đặt trong thư mục data_path)

**Command:**

```bash
DATA_PATH=[Your MSVD data and videos path]
python -m torch.distributed.launch --nproc_per_node=4 \
main_task_retrieval.py --do_train --num_thread_reader=2 \
--epochs=5 --batch_size=128 --n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/MSVD_Videos \
--output_dir ckpts/ckpt_msvd_retrieval_looseType \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype msvd \
--enriched yes \
--enriched_epochs=3 \
--enriched_max_steps=150 \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0 --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/32
```

**Các tham số mới:**
- `--enriched yes`: Kích hoạt enriched training cho MSVD
- `--enriched_epochs=3`: Số epochs train trên enriched data
- `--enriched_max_steps=150`: Số steps tối đa cho enriched training (dừng sớm nếu đạt)

**Nếu KHÔNG muốn dùng enriched training:**
```bash
# Bỏ tham số --enriched hoặc set --enriched no
```

---

### 2. MSRVTT Dataset

**Yêu cầu:**
- File enriched captions: `enriched_captions.json` (đường dẫn tùy chỉnh)
- File raw data: `MSRVTT_data.json` (truyền qua --data_path)

**Command:**

```bash
DATA_PATH=[Your MSRVTT data and videos path]
python -m torch.distributed.launch --nproc_per_node=4 \
main_task_retrieval.py --do_train --num_thread_reader=0 \
--epochs=5 --batch_size=128 --n_display=50 \
--train_csv ${DATA_PATH}/MSRVTT_train.9k.csv \
--val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/MSRVTT_data.json \
--enriched_data_path ${DATA_PATH}/enriched_captions.json \
--enriched_epochs=3 \
--enriched_max_steps=150 \
--features_path ${DATA_PATH}/MSRVTT_Videos \
--output_dir ckpts/ckpt_msrvtt_retrieval_looseType \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype msrvtt --expand_msrvtt_sentences  \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/32
```

**Các tham số mới:**
- `--enriched_data_path ${DATA_PATH}/enriched_captions.json`: Đường dẫn đến file enriched captions
- `--enriched_epochs=3`: Số epochs train trên enriched data
- `--enriched_max_steps=150`: Số steps tối đa cho enriched training

**Nếu KHÔNG muốn dùng enriched training:**
```bash
# Bỏ tham số --enriched_data_path
```

---

## Quy Trình Training

### Khi sử dụng enriched training:

1. **STAGE 1 - Enriched Data Training:**
   - Load enriched captions (enriched-caption-complete.pkl cho MSVD hoặc enriched_captions.json cho MSRVTT)
   - Train model trong `enriched_epochs` epochs hoặc cho đến khi đạt `enriched_max_steps`
   - Lưu checkpoint với prefix "enriched" (ví dụ: `pytorch_model.bin.enriched.0`)
   - Evaluate sau mỗi epoch và chọn checkpoint tốt nhất (R1 cao nhất)

2. **STAGE 2 - Original Data Fine-tuning:**
   - Load checkpoint tốt nhất từ STAGE 1
   - Load original captions (raw-captions.pkl cho MSVD hoặc MSRVTT_data.json cho MSRVTT)
   - Fine-tune model trong `epochs` epochs hoặc cho đến khi đạt `max_steps`
   - Lưu checkpoint thông thường (ví dụ: `pytorch_model.bin.0`)
   - Evaluate và chọn checkpoint tốt nhất

3. **Final Evaluation:**
   - Model tốt nhất từ STAGE 2 được sử dụng làm kết quả cuối cùng

### Khi KHÔNG sử dụng enriched training:

- Training diễn ra bình thường chỉ với original data
- Không có STAGE 1

---

## Output Files

**Với enriched training:**
```
ckpts/ckpt_msvd_retrieval_looseType/
├── pytorch_model.bin.enriched.0    # STAGE 1 - epoch 0
├── pytorch_model.bin.enriched.1    # STAGE 1 - epoch 1
├── pytorch_model.bin.enriched.2    # STAGE 1 - epoch 2
├── pytorch_model.bin.0             # STAGE 2 - epoch 0
├── pytorch_model.bin.1             # STAGE 2 - epoch 1
├── pytorch_model.bin.2             # STAGE 2 - epoch 2
├── pytorch_opt.bin.enriched.*      # Optimizer states cho STAGE 1
├── pytorch_opt.bin.*               # Optimizer states cho STAGE 2
└── log.txt                         # Training logs
```

**Logs sẽ hiển thị:**
```
==================================================
STAGE 1: Training on ENRICHED DATA
==================================================
Using enriched captions for MSVD
Enriched epochs: 3
Enriched max steps: 150
...
[ENRICHED] Epoch 1/3 Finished, Train Loss: 0.123
[ENRICHED] The best model is: ..., the R1 is: 45.20
...
==================================================
STAGE 1 COMPLETED: Best enriched model: ...
==================================================
==================================================
STAGE 2: Training on ORIGINAL DATA
==================================================
Using RAW captions: .../raw-captions.pkl
...
[ORIGINAL] Epoch 1/5 Finished, Train Loss: 0.098
[ORIGINAL] The best model is: ..., the R1 is: 48.50
```

---

## Lưu Ý Quan Trọng

1. **Early Stopping**: Nếu `enriched_max_steps` > 0, training sẽ dừng sớm khi đạt số steps đó (áp dụng cho cả STAGE 1 và STAGE 2 nếu set `max_steps`)

2. **File Paths**: 
   - MSVD: File enriched phải tên là `enriched-caption-complete.pkl` và nằm trong `data_path`
   - MSRVTT: File enriched path tùy chỉnh qua `--enriched_data_path`

3. **Checkpoints**: 
   - STAGE 1 lưu với prefix "enriched"
   - STAGE 2 lưu bình thường
   - Checkpoint tốt nhất từ STAGE 1 tự động được load vào STAGE 2

4. **Resume Training**: 
   - Hiện tại `--resume_model` chỉ áp dụng cho STAGE 2
   - Không hỗ trợ resume từ giữa STAGE 1

5. **Memory**: Training 2 giai đoạn sẽ tốn thời gian và bộ nhớ nhiều hơn. Đảm bảo có đủ disk space cho checkpoints.

---

## Troubleshooting

**Q: File enriched không tồn tại?**
- MSVD: Kiểm tra file `enriched-caption-complete.pkl` trong `data_path`
- MSRVTT: Kiểm tra đường dẫn `--enriched_data_path`
- Nếu không có file, training sẽ bỏ qua STAGE 1 và train trực tiếp trên original data

**Q: Muốn chỉ train STAGE 1 thôi?**
- Set `--epochs=0` để bỏ qua STAGE 2 (không khuyến nghị)

**Q: Muốn train lại STAGE 2 với checkpoint khác?**
- Sử dụng `--init_model` để chỉ định checkpoint cụ thể
- Bỏ `--enriched` hoặc `--enriched_data_path` để chỉ chạy training bình thường

**Q: Training bị dừng giữa chừng?**
- Kiểm tra logs xem có đạt `enriched_max_steps` hay `max_steps` chưa
- Kiểm tra GPU memory, có thể bị OOM

---

## Ví Dụ Chi Tiết

### Training MSVD với enriched data:
```bash
python -m torch.distributed.launch --nproc_per_node=4 \
main_task_retrieval.py --do_train --num_thread_reader=2 \
--epochs=5 --batch_size=128 --n_display=50 \
--data_path /path/to/MSVD \
--features_path /path/to/MSVD/MSVD_Videos \
--output_dir ckpts/ckpt_msvd_enriched \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype msvd \
--enriched yes \
--enriched_epochs=3 \
--enriched_max_steps=150 \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0 --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/32
```

### Training MSVD KHÔNG dùng enriched data:
```bash
python -m torch.distributed.launch --nproc_per_node=4 \
main_task_retrieval.py --do_train --num_thread_reader=2 \
--epochs=5 --batch_size=128 --n_display=50 \
--data_path /path/to/MSVD \
--features_path /path/to/MSVD/MSVD_Videos \
--output_dir ckpts/ckpt_msvd_baseline \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype msvd \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0 --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/32
```

Chỉ cần bỏ `--enriched yes` và các tham số liên quan!
