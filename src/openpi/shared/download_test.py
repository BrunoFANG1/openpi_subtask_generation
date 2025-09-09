import pathlib

import pytest

import openpi.shared.download as download


@pytest.fixture(scope="session", autouse=True)
def set_openpi_data_home(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("openpi_data")
    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("OPENPI_DATA_HOME", str(temp_dir))
        yield


def test_download_local(tmp_path: pathlib.Path):
    local_path = tmp_path / "local"
    local_path.touch()

    result = download.maybe_download(str(local_path))
    assert result == local_path

    with pytest.raises(FileNotFoundError):
        download.maybe_download("bogus")


def test_download_s3_dir():
    remote_path = "s3://openpi-assets/testdata/random"

    local_path = download.maybe_download(remote_path)
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path)
    assert new_local_path == local_path


def test_download_s3():
    remote_path = "s3://openpi-assets/testdata/random/random_512kb.bin"

    local_path = download.maybe_download(remote_path)
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path)
    assert new_local_path == local_path


def test_download_fsspec():
    remote_path = "gs://big_vision/paligemma_tokenizer.model"

    local_path = download.maybe_download(remote_path, gs={"token": "anon"})
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path, gs={"token": "anon"})
    assert new_local_path == local_path

def test_download_pi0_fast():
    remote_path = "s3://openpi-assets/checkpoints/pi0_fast_base/params"

    local_path = download.maybe_download(remote_path, gs={"token": "anon"})
    assert local_path.exists()

def test_download_pi05():
    remote_path = "gs://openpi-assets/checkpoints/pi05_base"

    local_path = download.maybe_download(remote_path, gs={"token": "anon"})
    assert local_path.exists()

def test_download_paligemma():
    remote_path = "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz"

    local_path = download.maybe_download(remote_path, gs={"token": "anon"})
    assert local_path.exists()

if __name__ == "__main__":
    # 自动下载太慢了，换香港专线下载
    import os
    os.environ["HTTP_PROXY"] = "http://10.7.145.219:3128"
    os.environ["HTTPS_PROXY"] = "http://10.7.145.219:3128"
    os.environ["OPENPI_DATA_HOME"] = "/x2robot_v2/xinyuanfang/projects_v2/.cache/openpi/test_data"
    test_download_pi05()
    print("Finished")