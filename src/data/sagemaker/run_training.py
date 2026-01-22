# src/sagemaker/run_training.py

import sagemaker
from sagemaker.pytorch import PyTorch

role = "arn:aws:iam::419154172513:role/SageMakerExecutionRole-BERT"
region = "ap-south-1"
bucket = "brt-ml-bucket-419154172513"

session = sagemaker.Session()

estimator = PyTorch(
    entry_point="train_sagemaker.py",
    source_dir="src",
    role=role,
    instance_type="ml.g4dn.xlarge",     # GPU (fast, affordable)
    instance_count=1,
    framework_version="2.0.1",
    py_version="py310",
    hyperparameters={},
    output_path=f"s3://{bucket}/models/",
    sagemaker_session=session
)

estimator.fit({
    "train": f"s3://{bucket}/gold/v2.1/train.csv",
    "val":   f"s3://{bucket}/gold/v2.1/val.csv"
})
