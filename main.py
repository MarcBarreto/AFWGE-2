import mlp
from afwge import AFWGE
from dataset import iris_preprocess
from utils import export_contrafactuals_custom_to_csv

def main():
    scaler, X, y, iris_df, train_loader, test_loader = iris_preprocess()

    model = mlp.train_model(X, y, train_loader, test_loader)

    print('---------------- Creating Contrafactuals ----------------')

    afwge = AFWGE(model, scaler, iris_df, select = 20, k = 1000)
    contrafactuals = afwge()

    export_contrafactuals_custom_to_csv(contrafactuals, iris_df)

    print('---------------- Finished ----------------')

if __name__ == '__main__':
    main()