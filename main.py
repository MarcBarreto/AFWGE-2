import mlp
from afwge import AFWGE
from dataset import iris_preprocess, pima_preprocess
from utils import export_counterfactuals_custom_to_csv

def main():
    aux = input()

    if aux == 'iris':
        scaler, X, y, dataset, train_loader, test_loader = iris_preprocess()
        partial_constraints = {}
    elif aux == 'pima':
        scaler, X, y, dataset, train_loader, test_loader = pima_preprocess()
        partial_constraints = {'Age': 'up'}
    else:
        print('Error: Choose iris or pima datasets')
        return
    
    model = mlp.train_model(X, y, train_loader, test_loader)

    print('---------------- Creating Counterfactuals ----------------')

    afwge = AFWGE(model, scaler, dataset, partial_constraints, select = 20, k = 1000)
    counterfactuals = afwge()

    export_counterfactuals_custom_to_csv(counterfactuals, dataset)

    print('---------------- Finished ----------------')

if __name__ == '__main__':
    main()