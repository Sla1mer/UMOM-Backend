# !pip install ultralytics - q
#
# !nvidia - smi
#
# import yaml
#
# with open('/content/dataset_yolo/data.yaml', 'r') as f:
#     config = yaml.safe_load(f)
# config['path'] = '/content/dataset_yolo'
# with open('/content/dataset_yolo/data.yaml', 'w') as f:
#     yaml.dump(config, f)
#
# from ultralytics import YOLO
# import shutil
# from google.colab import drive
# import os
#
# def save_to_drive_callback(trainer):
#     epoch = trainer.epoch
#     if epoch % 10 == 0:
#         dst = '/content/drive/MyDrive/yolo_weights_backup'
#         src = trainer.save_dir / 'weights'
#         if os.path.exists(dst):
#             shutil.rmtree(dst)
#         shutil.copytree(src, dst)
#         print(f"💾 Эпоха {epoch}: веса сохранены в Drive!")
#
# drive.mount('/content/drive')
#
# model = YOLO('yolo11m.pt')
#
# model.add_callback("on_train_epoch_end", save_to_drive_callback)
#
# results = model.train(
#     data='/content/dataset_yolo/data.yaml',
#     epochs=150,
#     imgsz=640,
#     batch=8,
#     device=0,
#     project='power_line_fast',
#     name='exp1',
#     patience=30,
#     save=True,
#
#     lr0=0.002,
#     lrf=0.0001,
#     warmup_epochs=3,
#     warmup_momentum=0.8,
#     warmup_bias_lr=0.01,
#
#     optimizer='AdamW',
#     momentum=0.937,
#     weight_decay=0.0005,
#
#     hsv_h=0.02,
#     hsv_s=0.8,
#     hsv_v=0.5,
#     degrees=12.0,
#     translate=0.15,
#     scale=0.6,
#     shear=2.5,
#     perspective=0.0,
#     flipud=0.0,
#     fliplr=0.5,
#     mosaic=1.0,
#     mixup=0.1,
#     copy_paste=0.0,
#
#     freeze=0,
#
#     close_mosaic=15,
#     amp=True,
#     save_period=10,
# )
#
# print("✅ Обучение завершено!")
#
# shutil.copy(
#     'power_line_fast/exp1/weights/best.pt',
#     '/content/drive/MyDrive/best_085_target.pt'
# )
#
# val_results = model.val(data='/content/dataset_yolo/data.yaml')
# print(f"\n🎯 Финальный mAP@50: {val_results.box.map50:.4f}")
# print(f"🎯 Финальный mAP@50-95: {val_results.box.map:.4f}")
#
# if val_results.box.map50 >= 0.85:
#     print("🎉 ЦЕЛЬ ДОСТИГНУТА! mAP@50 >= 0.85")
# else:
#     gap = 0.85 - val_results.box.map50
#     print(f"⚠️  До цели осталось: {gap:.4f} mAP")
