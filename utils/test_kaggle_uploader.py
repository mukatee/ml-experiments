__author__ = 'teemu kanstren'

import kaggle_uploader
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import os

def generate_synthetic_dataset(filepath):
    dirname = os.path.dirname(filepath)
    Path(dirname).mkdir(parents = True, exist_ok = True)
    mean = 0.5
    std = 0.1

    X = np.random.normal(mean, std, (395))
    Y = np.random.normal(mean * 2, std * 3, (395))
    synth_df = pd.DataFrame()
    synth_df["X"] = X
    synth_df["Y"] = Y
    synth_df.to_csv(filepath)

@pytest.fixture(autouse=True)
def run_around_tests():
    #before test comes here
    kaggle_uploader.reset()
    yield
    #after test comes here

def test_metadata_validate_empty(capsys):
    with pytest.raises(Exception) as e_info:
        kaggle_uploader.create_metadata()
    expected = """Invalid metadata: kaggle_uploader: base_path is not set, cannot proceed.
kaggle_uploader: user_id is not set, cannot proceed.
kaggle_uploader: dataset_id is not set, cannot proceed.
kaggle_uploader: title is not set, cannot proceed.
kaggle_uploader: no resource to upload defined, cannot proceed.
"""
    assert e_info.value.args[0] == expected
        #https://docs.pytest.org/en/latest/capture.html
        #captured = capsys.readouterr()
        #assert  captured.out == expected

def test_metadata_validate_invalid_slug(capsys):
    kaggle_uploader.dataset_id = "bad_name"
    #https://stackoverflow.com/questions/23337471/how-to-properly-assert-that-an-exception-gets-raised-in-pytest
    with pytest.raises(Exception) as e_info:
        kaggle_uploader.create_metadata()
    expected = """Invalid metadata: kaggle_uploader: base_path is not set, cannot proceed.
kaggle_uploader: user_id is not set, cannot proceed.
kaggle_uploader: dataset_id contains invalid chars. only a-zA-Z and - allowed. Cannot proceed.
kaggle_uploader: title is not set, cannot proceed.
kaggle_uploader: no resource to upload defined, cannot proceed.
"""
    assert e_info.value.args[0] == expected
    #https://docs.pytest.org/en/latest/capture.html
    #captured = capsys.readouterr()
    #assert  captured.out == expected

def test_metadata_no_basepath_resources():
    with pytest.raises(Exception) as e_info:
        kaggle_uploader.add_resource("test path", "test description")
    assert e_info.value.args[0] == "base_path must be set before adding resourcse. current: None"

def test_metadata_validate_valid_slug():
    kaggle_uploader.base_path = "base"
    kaggle_uploader.title = "hello title"
    kaggle_uploader.dataset_id = "alphanumeric-allowed-123"
    kaggle_uploader.title = "hello title"
    kaggle_uploader.user_id = "donkeys"
    kaggle_uploader.add_resource("test path", "test description")
    metadata = kaggle_uploader.create_metadata()
    assert metadata["id"] == "donkeys/alphanumeric-allowed-123"
    assert metadata["title"] == "hello title"
    assert len(metadata["resources"]) == 1

def test_data_upload():
    TEST_UPLOAD_DIR = "./test_upload_dir"
    TEST_FILENAME = "kaggle_test_data.csv"
    generate_synthetic_dataset(f"{TEST_UPLOAD_DIR}/{TEST_FILENAME}")
    kaggle_uploader.base_path = TEST_UPLOAD_DIR
    kaggle_uploader.title = "Kaggle uploader testset"
    kaggle_uploader.dataset_id = "kaggle-uploader-testset"
    kaggle_uploader.user_id = "donkeys"
    kaggle_uploader.add_resource(f"{TEST_FILENAME}", "test data")
#    kaggle_uploader.create()
    kaggle_uploader.update("new version")

