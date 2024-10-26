import sys
import mlp
import utils
from afwge import AFWGE
from dataset import iris_preprocess, pima_preprocess

def main():
    title = 'Dataset'
    if len(sys.argv) > 1:
        aux = sys.argv[1]
    else:
        aux = input("Type: iris to Iris dataset or pima to Pima Diabates Dataset") 

    if aux == 'iris':
        scaler, X, y, dataset, train_loader, test_loader = iris_preprocess()
        partial_constraints = {}
        title = 'Iris Dataset'
    elif aux == 'pima':
        path = input('Type: Path for Pima Diabete Dataset \n')
        scaler, X, y, dataset, train_loader, test_loader = pima_preprocess(path)
        partial_constraints = {'Age': 'up'}
        title = 'Pima Diabete Dataset'
    else:
        print('Error: Choose iris or pima datasets')
        return
    
    model = mlp.train_model(X, y, train_loader, test_loader)

    print('---------------- Creating Counterfactuals ----------------')

    afwge = AFWGE(model, scaler, dataset, partial_constraints, select = 400, k = 1000)
    counterfactuals = afwge()

    utils.export_counterfactuals_custom_to_csv(counterfactuals, dataset)
    print('---------------- Counterfactuals Created ----------------\n')
    print('---------------- Calculating Metrics ----------------\n')
    distance, distances = utils.calculate_distance(counterfactuals)
    print(f'Distances Average: {distance:.4f}\n')
    utils.box_plot(distances, f'Distance of Features on the {title}')
    print('---------------- Finished ----------------')

if __name__ == '__main__':
    main()