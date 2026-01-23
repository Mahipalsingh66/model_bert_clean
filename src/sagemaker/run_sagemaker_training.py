# import sagemaker
# from sagemaker.pytorch import PyTorch

# session = sagemaker.Session()
# role = sagemaker.get_execution_role()

# estimator = PyTorch(
#     entry_point="train_sagemaker.py",
#     source_dir="src/training",
#     role=role,
#     framework_version="2.0",
#     py_version="py310",
#     instance_count=1,
#     instance_type="ml.m5.large",   # cheap CPU
#     hyperparameters={},
#     output_path="s3://YOUR_BUCKET/models/test_run/",
# )

# estimator.fit({
#     "train": "s3://YOUR_BUCKET/gold/v2.1/"
# })
# run_sagemaker_training.py

import sagemaker
from sagemaker.pytorch import PyTorch

role = "arn:aws:iam::419154172513:role/SageMakerExecutionRole-BERT"

session = sagemaker.Session()

bucket = "brt-ml-bucket-419154172513"

estimator = PyTorch(
    entry_point="train_sagemaker.py",
    source_dir="src/training",
    role=role,
    framework_version="2.0.1",
    py_version="py39",
    instance_count=1,
    instance_type="ml.g4dn.xlarge",   # GPU (fast, cheap)
    hyperparameters={},
    output_path=f"s3://{bucket}/models/",
)

estimator.fit({
    "train": f"s3://{bucket}/gold/v2.1/train.csv",
    "val":   f"s3://{bucket}/gold/v2.1/val.csv"
})
