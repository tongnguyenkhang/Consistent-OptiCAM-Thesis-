### **PHASE 1: Train Compressed Model**

#### Train ResNet18 (Nhỏ hơn 4x)
python train_student_classifier.py --images_dir ".\images" --batch_size 32 --epochs 75 --lr 1e-6 --temperature 2.0 --optimizer adam 

### Sau khi KD xuống 0.03 thì đổi LR
python train_student_classifier.py --images_dir ".\images" --batch_size 32 --epochs 150 --lr 5e-6 --temperature 2.0 --optimizer adam --resume_from ".\student_resnet18_kd_finetuned_2.pth" --output ".\student_resnet18_kd_finetuned_3.pth"

### Xác nhận mức độ giống nhau vector xác suất trên cả dataset
python eval_student_vs_teacher.py 
  --images_dir ".\images" 
  --student_ckpt ".\student_resnet18_kd_finetuned_3.pth" 
  --batch_size 32 
  --temperature 2.0


Tổng cộng Epoch: 550 epochs.
---

### **PHASE 2: Run OptiCAM Multi with Compressed Model**

#### Step 2.1: Run BASELINE (Teacher - ResNet50)
```bash
# First, establish baseline with original ResNet50
python generate_opticam_multi.py --name_path OptiCAM_Multi_Uncompressed --learning_rate 1e-6 --max_iter 200 --batch_size 5 --num_masks 3 --use_lambda_scheduling --lambda_start 1.0 --lambda_end 0.3 --use_mixed_precision --viz_from_combined
```


#### Step 2.2: Run COMPRESSED (Student - ResNet18)
```bash
# With compressed ResNet18 student model
python generate_opticam_multi_compressed.py --student_path ./models_test/student_resnet18_kd_finetuned_6.pth --student_arch resnet18 --name_path OptiCAM_Multi_Compressed_ResNet18 --learning_rate 1e-6 --max_iter 200 --batch_size 5 --num_masks 3 --use_lambda_scheduling --lambda_start 1.0 --lambda_end 0.3 --use_mixed_precision --viz_from_combined


```
**Expected speedup:** ~3-4x faster 
---

### **PHASE 3: Compare Results**

```bash
python compare_teacher_student.py --teacher_dir ./results/OptiCAM_Multi_Uncompressed --student_dir ./results/OptiCAM_Multi_Compressed_ResNet18 --output comparison_resnet18.txt

```

**Output:** Comprehensive comparison report with:
- Primary metrics (AD/AI/AG) differences
- Advanced metrics (AUC/AOPC) differences
- Speed improvements
- Quality vs Speed trade-off analysis