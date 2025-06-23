import os
os.environ["HF_LEROBOT_HOME"] = "/x2robot/xinyuanfang/projects/.cache/lerobot"
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
os.environ['OPENPI_DATA_HOME'] = '/x2robot/xinyuanfang/projects/.cache/openpi'

from openpi.shared.download import maybe_download, _download_boto3

if __name__ == "__main__":
    # path = maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    _download_boto3("s3://openpi-assets/checkpoints/pi0_base/params", local_path="/x2robot/xinyuanfang/projects/.cache/openpi/openpi-assets/checkpoints/pi0_base/params")