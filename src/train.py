import multiprocessing
import pandas as pd
from config import Config
from model import TextClassifier
from utils import set_seed

def main():
    cfg = Config()
    set_seed(cfg.SEED)
    df = pd.read_csv("data/combined_news_dataset_V2.csv", sep=",")
    classifier = TextClassifier(cfg)
    classifier.prepare_data(df)
    classifier.build_model()
    classifier.train()
    # После тренировки — финальная оценка на test set
    print("Final evaluation on test set:")
    classifier.evaluate(loader=classifier.test_loader)
    classifier.save()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()