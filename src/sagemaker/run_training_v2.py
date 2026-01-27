# # ============================================================
# # SAGEMAKER LAUNCH SCRIPT — ENTERPRISE TRAINING + HPO READY
# # File    : run_training_v2.py
# # Owner   : Mahipal Singh
# # Purpose : Launch baseline + manual tuning + HPO safely
# # Region  : ap-south-1
# # ============================================================

# import sagemaker
# from sagemaker.pytorch import PyTorch
# from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, CategoricalParameter

# # ------------------------------------------------------------
# # AWS DETAILS (FIXED FOR YOUR ACCOUNT)
# # ------------------------------------------------------------

# role   = "arn:aws:iam::419154172513:role/SageMakerExecutionRole-BERT"
# bucket = "brt-ml-bucket-419154172513"
# region = "ap-south-1"

# session = sagemaker.Session()

# # ------------------------------------------------------------
# # BASE ESTIMATOR — USED FOR MANUAL + HPO
# # ------------------------------------------------------------

# estimator = PyTorch(
#     entry_point="train_sagemaker_v2.py",   # <— NEW TRAINING SCRIPT

#     # Upload only this folder (contains train_sagemaker_v2.py)
#     source_dir="D:/model_bert_copy/src/sagemaker",

#     role=role,
#     instance_type="ml.g4dn.xlarge",   # Best GPU / cost ratio
#     instance_count=1,

#     framework_version="2.0.1",
#     py_version="py310",

#     output_path=f"s3://{bucket}/models/",

#     # Cost control & stability
#     disable_profiler=True,
#     debugger_hook_config=False,

#     sagemaker_session=session
# )

# # ============================================================
# # PHASE 1 — BASELINE TRAINING (RUN THIS FIRST)
# # ============================================================

# baseline_hyperparameters = {
#     "epochs": 3,
#     "batch_size": 16,
#     "learning_rate": 2e-5,
#     "weight_decay": 0.01,
#     "warmup_ratio": 0.1,
#     "use_class_weights": 0,
#     "seed": 42
# }

# estimator.set_hyperparameters(**baseline_hyperparameters)

# print("\n🚀 Launching BASELINE training job...\n")

# estimator.fit({
#     "train": f"s3://{bucket}/gold/v2.3/train.csv",
#     "val":   f"s3://{bucket}/gold/v2.3/val.csv"
# })

# # ============================================================
# # PHASE 2 — MANUAL TUNING (UNCOMMENT WHEN NEEDED)
# # ============================================================

# """
# # Example manual sweep (learning rate test)

# manual_hyperparameters = {
#     "epochs": 3,
#     "batch_size": 16,
#     "learning_rate": 3e-5,   # change this manually
#     "weight_decay": 0.01,
#     "warmup_ratio": 0.1,
#     "use_class_weights": 0,
#     "seed": 42
# }

# estimator.set_hyperparameters(**manual_hyperparameters)

# print("\n🚀 Launching MANUAL tuning job...\n")

# estimator.fit({
#     "train": f"s3://{bucket}/gold/v2.3/train.csv",
#     "val":   f"s3://{bucket}/gold/v2.3/val.csv"
# })
# """

# # ============================================================
# # PHASE 3 — AUTOMATIC HYPERPARAMETER TUNING (FINAL POLISH)
# # ============================================================

# """
# # Uncomment only AFTER manual tuning is complete

# hyperparameter_ranges = {
#     "learning_rate": ContinuousParameter(1e-5, 4e-5),
#     "batch_size": CategoricalParameter([16, 32]),
#     "weight_decay": ContinuousParameter(0.0, 0.05),
#     "neu_weight": ContinuousParameter(1.1, 1.6),
#     "warmup_ratio": ContinuousParameter(0.05, 0.12),
# }

# # Fixed production settings
# estimator.set_hyperparameters(
#     epochs=3,
#     use_class_weights=1,
#     neg_weight=1.0,
#     pos_weight=1.0,
#     seed=42
# )

# tuner = HyperparameterTuner(
#     estimator=estimator,
#     objective_metric_name="validation:macro_f1",
#     hyperparameter_ranges=hyperparameter_ranges,
#     objective_type="Maximize",
#     max_jobs=20,
#     max_parallel_jobs=2,
# )

# print("\n🔥 Launching AUTOMATIC HPO job...\n")

# tuner.fit({
#     "train": f"s3://{bucket}/gold/v2.3/train.csv",
#     "val":   f"s3://{bucket}/gold/v2.3/val.csv"
# })
# """

# # ============================================================
# # END OF RUN SCRIPT
# # ============================================================
# ============================================================
# SAGEMAKER LAUNCH SCRIPT — CX BACKBONE PHASE-1 (COST SAFE)
# File    : run_training_v2.py
# Owner   : Mahipal Singh
# Purpose : Launch ONE multi-head training (sentiment + intent + aspect)
# Region  : ap-south-1
# ============================================================

import sagemaker
from sagemaker.pytorch import PyTorch

# ------------------------------------------------------------
# AWS DETAILS (FIXED)
# ------------------------------------------------------------

role   = "arn:aws:iam::419154172513:role/SageMakerExecutionRole-BERT"
bucket = "brt-ml-bucket-419154172513"
region = "ap-south-1"

session = sagemaker.Session()

# ------------------------------------------------------------
# ESTIMATOR — SINGLE COST-SAFE RUN
# ------------------------------------------------------------

estimator = PyTorch(
    entry_point="train_sagemaker_v2.py",   # multi-head trainer

    # Upload only sagemaker folder
    source_dir="D:/model_bert_copy/src/sagemaker",

    role=role,
    instance_type="ml.g4dn.xlarge",   # best GPU / cost ratio
    instance_count=1,

    framework_version="2.0.1",
    py_version="py310",

    output_path=f"s3://{bucket}/models/",

    # Cost control
    disable_profiler=True,
    debugger_hook_config=False,

    # 🔥 FINAL PRODUCTION HYPERPARAMETERS
    hyperparameters={
        "EPOCHS": 3,
        "BATCH_SIZE": 16,
        "LR": 2e-5,
        "MAX_LENGTH": 160,
        "MODEL_NAME": "xlm-roberta-base"
    },

    sagemaker_session=session
)

# ------------------------------------------------------------
# FINAL CX PHASE-1 DATA (MULTI-OUTPUT)
# ------------------------------------------------------------

print("\n🚀 Launching CX Backbone v1 training (ONE RUN ONLY)...\n")

estimator.fit({
    "train": f"s3://{bucket}/gold/v2.3_multitask/train_cx_v1.csv",
    "val":   f"s3://{bucket}/gold/v2.3_multitask/val_multi_v1.csv"
})

# ============================================================
# END — NO HPO, NO MANUAL SWEEPS (COST SAFE DESIGN)
# ============================================================
