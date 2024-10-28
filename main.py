import sys
import mlp
import utils
from afwge import AFWGE
import dataset

def main():
    title = 'Dataset'
    if len(sys.argv) > 1:
        aux = sys.argv[1]
    else:
        aux = input("Type: iris to Iris dataset; pima to Pima Diabates Dataset; breast to Breast Cancer Dataset or adult to Adult Income Dataset") 

    encoded = False
    encoded_columns = []
    constraints = []
    partial_constraints = {}
    
    if aux == 'iris':
        scaler, X, y, df, train_loader, test_loader = dataset.iris_preprocess()
        title = 'Iris Dataset'
    elif aux == 'pima':
        path = input('Type: Path for Pima Diabete Dataset \n')
        scaler, X, y, df, train_loader, test_loader = dataset.pima_preprocess(path)
        partial_constraints = {'Age': 'up'}
        title = 'Pima Diabete Dataset'
    elif aux == 'breast':
        scaler, X, y, df, train_loader, test_loader = dataset.breast_preprocess()
        title = 'Breast Cancer Dataset'
    elif aux == 'adult':
        path = input('Type: Path for Pima Diabete Dataset \n')
        scaler, X, y, df, train_loader, test_loader, encoded_columns = dataset.adult_preprocess(path)
        encoded = True
        constraints = ['Race', 'Sex']
        partial_constraints = {'Age': 'up'}
    else:
        print('Error: Choose iris, pima, breast or adult datasets')
        return
    
    model = mlp.train_model(X, y, train_loader, test_loader)

    print('---------------- Creating Counterfactuals ----------------')

    afwge = AFWGE(model, scaler, df, constraints, partial_constraints,select = 400, k = 1000, encoded = encoded, encoded_columns = encoded_columns)
    counterfactuals = afwge()

    utils.export_counterfactuals_custom_to_csv(counterfactuals, df)
    print('---------------- Counterfactuals Created ----------------\n')
    print('---------------- Calculating Metrics ----------------\n')
    distance, distances, changed_features = utils.calculate_metrics(counterfactuals)
    print(f'Distances Average: {distance:.4f}\n')
    print(f'Number of features Changed: {changed_features}\n')
    utils.box_plot(distances, f'Distance of Features on the {title}')
    print('---------------- Finished ----------------')

if __name__ == '__main__':
    main()