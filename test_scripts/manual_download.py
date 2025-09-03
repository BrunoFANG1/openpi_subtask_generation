import os
os.environ["HF_LEROBOT_HOME"] = "/x2robot_v2/xinyuanfang/projects_v2/.cache/lerobot"
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
os.environ['OPENPI_DATA_HOME'] = '/x2robot_v2/xinyuanfang/projects_v2/.cache/openpi'

from openpi.shared.download import maybe_download, _download_boto3

if __name__ == "__main__":
    path = maybe_download("s3://openpi-assets-preview/checkpoints/pi05_may21_280k_v1/params")
    # _download_boto3("gs://openpi-assets/checkpoints/pi0_fast_base/params", local_path="/x2robot/xinyuanfang/projects/.cache/openpi/openpi-assets/checkpoints/pi05_test")