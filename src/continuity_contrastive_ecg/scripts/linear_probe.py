import os

import hydra
import rootutils
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

os.environ["PROJECT_ROOT"] = str(rootutils.setup_root(search_from=__file__, indicator=".project-root"))


@hydra.main(version_base=None, config_path="configs", config_name="linear_probe")
def main(cfg: DictConfig):
    datasets = instantiate(cfg.data)
    model = instantiate(cfg.model)
    ckpt = torch.load(cfg.pretrain_ckpt_path)
    model.load_state_dict(ckpt["state_dict"])
    model.freeze()

    # check if the model is supervised or contrastive
    if hasattr(model, "get_predictions"):
        # calculate the accuracy and AUC of the model from the predictions
        train_preds = []
        train_labels = []
        for batch in datasets.train_loader:
            x, y = batch
            preds = model.get_predictions(x)
            train_preds.append(preds)
            train_labels.append(y)
        train_preds = torch.cat(train_preds, dim=0)
        train_labels = torch.cat(train_labels, dim=0)

        test_preds = []
        test_labels = []
        for batch in datasets.test_loader:
            x, y = batch
            preds = model.get_predictions(x)
            test_preds.append(preds)
            test_labels.append(y)
        test_preds = torch.cat(test_preds, dim=0)
        test_labels = torch.cat(test_labels, dim=0)

        # calculate the accuracy
        correct = (test_preds.argmax(dim=1) == test_labels).sum().item()
        acc = correct / len(test_labels)
        print(f"Accuracy: {acc}")
        # AUROC
        y_pred = test_preds
        auroc = roc_auc_score(test_labels, y_pred)
        print(f"AUROC: {auroc}")

    elif hasattr(model, "get_representations"):
        # run a logistic regression on the representations
        # get the representations
        train_reps = []
        train_labels = []
        for batch in datasets.train_loader:
            x, y = batch
            reps = model.get_representations(x)
            train_reps.append(reps)
            train_labels.append(y)
        train_reps = torch.cat(train_reps, dim=0)
        train_labels = torch.cat(train_labels, dim=0)

        test_reps = []
        test_labels = []
        for batch in datasets.test_loader:
            x, y = batch
            reps = model.get_representations(x)
            test_reps.append(reps)
            test_labels.append(y)
        test_reps = torch.cat(test_reps, dim=0)
        test_labels = torch.cat(test_labels, dim=0)

        # fit a logistic regression model
        clf = LogisticRegression(random_state=0, max_iter=1000).fit(train_reps, train_labels)
        acc = clf.score(test_reps, test_labels)
        print(f"Accuracy: {acc}")
        # AUROC
        y_pred = clf.predict_proba(test_reps)[:, 1]
        auroc = roc_auc_score(test_labels, y_pred)
        print(f"AUROC: {auroc}")

    else:
        raise ValueError("Model does not have a get_predictions or get_representations method.")

    return model.min_val_loss


if __name__ == "__main__":
    main()
