import mlp
from afwge import AFWGE
from dataset import iris_preprocess
from utils import export_counterfactuals_custom_to_csv

def main():
    scaler, X, y, iris_df, train_loader, test_loader = iris_preprocess()

    model = mlp.train_model(X, y, train_loader, test_loader)

    print('---------------- Creating Counterfactuals ----------------')

    afwge = AFWGE(model, scaler, iris_df, select = 20, k = 1000)
    counterfactuals = afwge()

    export_counterfactuals_custom_to_csv(counterfactuals, iris_df)

    print('---------------- Finished ----------------')

if __name__ == '__main__':
    main()