from openpi.shared.download import maybe_download, _download_boto3

if __name__ == "__main__":
    _download_boto3("s3://openpi-assets/checkpoints/pi0_base/params", local_path="/x2robot/xinyuanfang/projects/openpi/checkpoints/baseckpt")