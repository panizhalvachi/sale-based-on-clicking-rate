import mlflow
import torch
from torch.utils.data import RandomSampler, DataLoader

import datasets
import feature_enginier
import modules
import utils
import pandas as pd


class Pipeline:
    @staticmethod
    def run(config):
        print("start pipe line")
        Pipeline.get_data(config)
        print("finish getting data/ start preprocessing data")
        Pipeline.preprocess_data(config)
        print("finish preprocessing data/ start training")
        Pipeline.train(config)
        print("finish training/ start evaluation")
        Pipeline.evaluation(config)
        print("finish evaluation/ start deploy")
        Pipeline.deploy(config)

    @staticmethod
    def deploy(config):
        train_data = pd.read_pickle(config['train pure address'])
        feature_builder = feature_enginier.feature_engineering(data=train_data)
        feature_builder.clean_function()
        model = Pipeline.load_model(config)
        model.eval()
        with torch.no_grad():
            while True:
                inp = input("test dataset address:")
                test_data = pd.read_pickle(inp)
                feature_builder.change_data(test_data)
                feature_builder.clean_function()

                dataset = datasets.RecomenderTestingDataset(test_data)
                dataloader = DataLoader(
                    dataset,
                    batch_size=1,
                    num_workers=0,
                    drop_last=False,
                    pin_memory=True)
                outputs = []
                for step, x in enumerate(dataloader):
                    # forward pass
                    x = x.to(config['device'])
                    output = model(x)
                    outputs.append(output)

                outputs = (torch.cat(outputs) > 0.5).squeeze().numpy()
                print(outputs)
                output = pd.DataFrame(outputs, columns=["predict"])
                output.to_csv("data/prediction.csv")

    @staticmethod
    def get_data(config):
        with mlflow.start_run(config["run id"]):
            all_data = pd.read_csv(config['data address'])
            n = len(all_data)

            # set dataset sizes / train_set:80%, validation_set:10%, test_set:10%
            train_size = int(n * 0.80)
            dev_size = int(n * 0.10)
            test_size = n - train_size - dev_size

            # build datasets
            train_data = all_data.iloc[:train_size]
            dev_data = all_data.iloc[train_size:train_size + dev_size]
            test_data = all_data.iloc[-test_size:]

            train_data.to_pickle(config['train pure address'])
            dev_data.to_pickle(config['dev pure address'])
            test_data.to_pickle(config['test pure address'])

            mlflow.log_artifact(config['train pure address'])
            mlflow.log_artifact(config['dev pure address'])
            mlflow.log_artifact(config['test pure address'])

    @staticmethod
    def preprocess_data(config):
        with mlflow.start_run(config["run id"]):
            train_data = pd.read_pickle(config['train pure address'])
            dev_data = pd.read_pickle(config['dev pure address'])
            test_data = pd.read_pickle(config['test pure address'])
            feature_builder = feature_enginier.feature_engineering(data=train_data)
            feature_builder.clean_function()
            feature_builder.change_data(dev_data, train=False)
            feature_builder.clean_function()
            feature_builder.change_data(test_data, train=False)
            feature_builder.clean_function()

            train_data.to_pickle(config['train clean address'])
            dev_data.to_pickle(config['dev clean address'])
            test_data.to_pickle(config['test clean address'])

            mlflow.log_artifact(config['train clean address'])
            mlflow.log_artifact(config['dev clean address'])
            mlflow.log_artifact(config['test clean address'])

    @staticmethod
    def train(config):
        model_dict = {
            "wide deep": modules.WideDeepModel,
            "res wide deep": modules.ResWideDeepModel,
            "linear": modules.LinearModel,
            "constant": modules.BaseLine
        }
        model_class = model_dict[config['model_name']]
        model = model_class([4, 4, 6, 22, 8], 5).to(config['device'])

        epochs = config['epochs']
        lr = config['lr']
        batch_size = config['batch_size']
        device = config['device']

        class_weight = torch.tensor([0.13, 0.87])
        train_data = pd.read_pickle(config['train clean address'])
        dev_data = pd.read_pickle(config['dev clean address'])
        utils.train_and_eval_model(epochs, lr, batch_size, device, model, class_weight,
                                   train_data, dev_data, print_res=False, run_id=config['run id'])
        Pipeline.save_model(config, model)

    @staticmethod
    def evaluation(config):
        model = Pipeline.load_model(config)
        batch_size = config['batch_size']
        test_data = pd.read_pickle(config['test clean address'])
        f1, precision, recall = utils.eval_model(0, model,
                                                 utils.build_dataloader(test_data, batch_size),
                                                 print_res=False, log_result=False)
        print(f" f1 score on test={f1}")

    @staticmethod
    def load_model(config):
        model_uri = "runs:/{}/model".format(config["run id"])
        model = mlflow.pytorch.load_model(model_uri)
        return model

    @staticmethod
    def save_model(config, model):
        with mlflow.start_run(config["run id"]):
            mlflow.pytorch.log_model(model, "model")
