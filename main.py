import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
import pipeline

pipe = pipeline.Pipeline()
with mlflow.start_run() as run:
    config = {
        "epochs": 1,
        "lr": 0.001,
        "batch_size": 4000,
        "device": "cpu",
        "model_name": "res wide deep",
        "run id": run.info.run_id,

        'data address': "train_dataset.csv",
        'train pure address': "data/train_pure_dataset.pickle",
        'dev pure address': "data/dev_pure_dataset.pickle",
        'test pure address': "data/test_pure_dataset.pickle",
        'train clean address': "data/train_clean_dataset.pickle",
        'dev clean address': "data/dev_clean_dataset.pickle",
        'test clean address': "data/test_clean_dataset.pickle",
    }
mlflow.end_run()
pipe.run(config)