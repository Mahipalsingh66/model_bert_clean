# ============================================================
# SAGEMAKER LAUNCH SCRIPT — MULTI-TASK TRAINING
# File: run_training_multitask_banking.py
# ============================================================

import sagemaker
from sagemaker.pytorch import PyTorch

role   = "arn:aws:iam::419154172513:role/SageMakerExecutionRole-BERT"
bucket = "brt-ml-bucket-419154172513"

session = sagemaker.Session()

estimator = PyTorch(
    entry_point="train_sagemaker_multitask_banking.py",   # ✅ banking trainer
    source_dir="D:/model_bert_copy/src/sagemaker",
    role=role,
    instance_type="ml.g4dn.xlarge",
    instance_count=1,
    framework_version="2.0.1",
    py_version="py310",
    output_path=f"s3://{bucket}/models/banking/",         # ✅ isolate banking models
    disable_profiler=True,
    debugger_hook_config=False,
    hyperparameters={
        "EPOCHS": 5,
        "BATCH_SIZE": 16,
        "LR": 1e-5
    },
    sagemaker_session=session
)

estimator.fit({
    "train": f"s3://{bucket}/gold/banking_data_final/train.csv",  # ✅ banking data
    "val":   f"s3://{bucket}/gold/banking_data_final/val.csv"
})