# BIRD: Behavior Induction via Representation-structure Distillation

This repository provides an implementation of BIRD (Behavior Induction via Representation-structure Distillation), a framework for transferring aligned behavioral properties to student models via representational supervision. This implementation demonstrates BIRD in the context of robust behavior transfer for image classification, where a student model inherits robustness to image corruptions from an aligned teacher model.

In the sections below, we provide instructions for the following workflow:
1. Data Preparation: Generate clean and corrupted datasets for teacher training and student evaluation using standard image classification benchmarks and ImageNet-C-style corruptions.
2. Train an Aligned Teacher: Train a teacher model on a source dataset with robustness-enhancing augmentations.
3. Pretrain a Student Model: Train a student on a separate (target) dataset using standard supervised learning, without robustness augmentations.
4. Behavior Transfer via BIRD: Fine-tune the student to match the representational structure of the teacher while preserving original task performance, using a combined task + representation-structre based loss.
5. Evaluation: Evaluate model robustness on held-out corruptions to quantify behavior transfer.


## Getting Started

### 0. Installation
Install dependencies and set up the environment with.  We recommend configuring your environment with python version 3.10.
```
pip install .
```

### 1. Create corrupted datasets
To create augmented variants of CIFAR-10 and CIFAR-100 training and test datasets, run the following:
```
python preprocess_data/make_cifar_c.py --save_dir ./data
```

### 2. Train a robust teacher model
Pre-train teacher model on clean source data:
```
python pretrain.py --dataset cifar10 --dataset_root ./data/cifar10 --model_name <choice_of_teacher_model_arch> --ckpt_dir ./models/pretrain_teacher
```

Tune teacher model on corrupted source data:
```
python pretrain.py --dataset cifar10-c --dataset_root ./data/cifar10-c --model_name <choice_of_teacher_model_arch> --pretrained_model_path ./models/pretrain_teacher/best_acc.pt --ckpt_dir ./models/robust_teacher
```

Currently supported options for model choices include: `alexnet`, `densenet121`, `densenet169`, `densenet201`, `mobilenetv2`, `resnet18`, `resnet34`, `resnet50`, `resnet101`,  and `resnet152`.

In the commands of this step (and the steps that follow), dataset and model checkpoint paths can be changed accoridng to where you want data and models saved locally.

### 3. Pre-train a non-robust student model
```
python pretrain.py --dataset cifar100 --dataset_root ./data/cifar100 --model_name <choice_of_student_model_arch> --ckpt_dir ./models/pretrain_student
```

## Robust behavior transfer with BIRD

### 4. Transfer robustness from teacher to student with BIRD
```
python distill_behavior.py --student_dataset cifar100 --student_dataset_root ./data/cifar100 --teacher_dataset cifar10 --student_model_name <choice_of_student_model_arch> --teacher_model_name <choice_of_teacher_model_arch> --pretrained_student_model_path ./models/pretrain_student/best_acc.pt --pretrained_teacher_model_path ./models/robust_teacher/best_acc.pt --student_alignment_layer <student_layer_to_guide> --teacher_alignment_layer <teacher_guiding_layer> --ckpt_dir ./models/bird_student
```

### 5. Evaluate BIRD-tuned student performance
Evaluation on corrupt test data:
```
python evaluate.py --dataset cifar100-c --dataset_root ./data/cifar100-c --model_name <choice_of_student_model_arch> --trained_model_path ./models/bird_student/best_loss.pt
```

Evaluation on clean test data:
```
python evaluate.py --dataset cifar100 --dataset_root ./data/cifar100 --model_name <choice_of_student_model_arch> --trained_model_path ./models/bird_student/best_loss.pt
```
