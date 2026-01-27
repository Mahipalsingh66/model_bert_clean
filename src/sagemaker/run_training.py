# # src/sagemaker/run_training.py

# import sagemaker
# from sagemaker.pytorch import PyTorch

# role = "arn:aws:iam::419154172513:role/SageMakerExecutionRole-BERT"
# region = "ap-south-1"
# bucket = "brt-ml-bucket-419154172513"

# session = sagemaker.Session()

# estimator = PyTorch(
#     entry_point="train_sagemaker.py",
#     source_dir="D:/model_bert_copy/src/data/sagemaker",
#     role=role,
#     instance_type="ml.g4dn.xlarge",     # GPU (fast, affordable)
#     instance_count=1,
#     framework_version="2.0.1",
#     py_version="py310",
#     hyperparameters={},
#     output_path=f"s3://{bucket}/models/",
#     sagemaker_session=session
# )

# estimator.fit({
#     "train": f"s3://{bucket}/gold/v2.1/train.csv",
#     "val":   f"s3://{bucket}/gold/v2.1/val.csv"
# })
import sagemaker
from sagemaker.pytorch import PyTorch

# AWS details
role = "arn:aws:iam::419154172513:role/SageMakerExecutionRole-BERT"
bucket = "brt-ml-bucket-419154172513"

session = sagemaker.Session()

estimator = PyTorch(
    entry_point="train_sagemaker.py",

    # Upload only this folder
    source_dir="D:/model_bert_copy/src/sagemaker",

    role=role,
    instance_type="ml.g4dn.xlarge",
    instance_count=1,

    framework_version="2.0.1",
    py_version="py310",

    output_path=f"s3://{bucket}/models/",

    # Cost control
    disable_profiler=True,
    debugger_hook_config=False,

    sagemaker_session=session
)

# 🔥 v2.2 DATA
estimator.fit({
    "train": f"s3://{bucket}/gold/v2.2/train.csv",
    "val":   f"s3://{bucket}/gold/v2.2/val.csv"
})
