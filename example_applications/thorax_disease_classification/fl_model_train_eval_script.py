
import data_utils
import fl_utils



if __name__ == '__main__':

    num_clients = 100
    batch_size = 32
    num_workers = 1
    fraction_fit = 0.1
    fraction_evaluate = 0.2
    min_fit_clients = 10
    min_evaluate_clients = 20
    num_rounds = 10

    trainloaders, valloaders, testloader = data_utils.get_fl_nih_subset(
        num_clients = num_clients,
        batch_size = batch_size
        )

    

    print(len(trainloaders))

    


