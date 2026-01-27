# ============================================================
# FILE : run_training_phase4_v2.py
# PURPOSE: Launch Phase-4 FINAL Aspect Sentiment (v2)
# ============================================================

import sagemaker
from sagemaker.pytorch import PyTorch

role   = "arn:aws:iam::419154172513:role/SageMakerExecutionRole-BERT"
bucket = "brt-ml-bucket-419154172513"

session = sagemaker.Session()

estimator = PyTorch(
    entry_point="train_sagemaker_phase4_v2.py",
    source_dir="D:/model_bert_copy/src/sagemaker",

    role=role,
    instance_type="ml.g4dn.xlarge",
    instance_count=1,

    framework_version="2.0.1",
    py_version="py310",

    output_path=f"s3://{bucket}/models/",

    disable_profiler=True,
    debugger_hook_config=False,

    hyperparameters={
        "EPOCHS": 4,
        "BATCH_SIZE": 16,
        "LR": 2e-5,
        "MAX_LENGTH": 160,
        "MODEL_NAME": "xlm-roberta-base"
    },

    sagemaker_session=session
)

print("\n🚀 Launching Phase-4 FINAL (v2) training...\n")

estimator.fit({
    "train": f"s3://{bucket}/gold/cx_phase4/train.csv",
    "val":   f"s3://{bucket}/gold/cx_phase4/val.csv"
})
