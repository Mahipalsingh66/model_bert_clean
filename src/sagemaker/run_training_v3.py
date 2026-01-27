
import sagemaker
from sagemaker.pytorch import PyTorch

role   = "arn:aws:iam::419154172513:role/SageMakerExecutionRole-BERT"
bucket = "brt-ml-bucket-419154172513"

session = sagemaker.Session()

estimator = PyTorch(
    entry_point="train_sagemaker_v3.py",
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
        "EPOCHS": 3,
        "BATCH_SIZE": 16,
        "LR": 2e-5,
        "MAX_LENGTH": 160,
        "MODEL_NAME": "xlm-roberta-base"
    },
    sagemaker_session=session
)

print("\n🚀 Launching CX Phase-2 training (Sentiment + Intent)...\n")

estimator.fit({
    "train": f"s3://{bucket}/gold/v2.3_multitask/train_cx_v1.csv",
    "val":   f"s3://{bucket}/gold/v2.3_multitask/val_multi_v1.csv"
})

