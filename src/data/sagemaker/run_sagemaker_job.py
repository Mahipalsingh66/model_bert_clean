import sagemaker
from sagemaker.pytorch import PyTorch

session = sagemaker.Session()
role = sagemaker.get_execution_role()

estimator = PyTorch(
    entry_point="train_sagemaker.py",
    source_dir="src/training",
    role=role,
    framework_version="2.0",
    py_version="py310",
    instance_count=1,
    instance_type="ml.m5.large",   # cheap CPU
    hyperparameters={},
    output_path="s3://YOUR_BUCKET/models/test_run/",
)

estimator.fit({
    "train": "s3://YOUR_BUCKET/gold/v2.1/"
})
