# ============================================================
# SAGEMAKER LAUNCH SCRIPT — MULTI-TASK INDUSTRY MODEL (FINAL)
# File: run_training_multitask_v2.py
# ============================================================

import sagemaker
from sagemaker.pytorch import PyTorch

# -------------------------
# AWS CONFIG
# -------------------------
ROLE   = "arn:aws:iam::419154172513:role/SageMakerExecutionRole-BERT"
BUCKET = "brt-ml-bucket-419154172513"
# REGION = "ap-south-1"

session = sagemaker.Session()
# sagemaker.session.Session(boto_region_name=REGION)

# -------------------------
# ESTIMATOR
# -------------------------
estimator = PyTorch(
    entry_point="train_sagemaker_multitask_v2.py",
    source_dir="D:/model_bert_copy/src/sagemaker",   # must contain training file
    role=ROLE,
    instance_type="ml.g4dn.xlarge",
    instance_count=1,
    framework_version="2.0.1",
    py_version="py310",
    output_path=f"s3://{BUCKET}/models/",
    disable_profiler=True,
    debugger_hook_config=False,
    hyperparameters={
        "EPOCHS": 3,          # 🔥 test run
        "BATCH_SIZE": 16,
        "LR": 2e-5
    },
    sagemaker_session=session
)

# -------------------------
# TRAINING DATA (TEST SET)
# -------------------------
estimator.fit({
    "train": f"s3://{BUCKET}/gold/gold_test_data/train.csv",
    "val":   f"s3://{BUCKET}/gold/gold_test_data/val.csv"
})