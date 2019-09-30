import torch
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
<<<<<<< HEAD
from torch.utils.data.sampler import SequentialSampler
=======
>>>>>>> e3866c46f4ea3e48390a009cf47add22fe43551a
import numpy as np

import matplotlib.pyplot as plt


def DatasetSplit(dataset, test_split, shuffle_dataset, random_seed, batchsize, testBatchsize, numworkers, pinmemory):

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
<<<<<<< HEAD
    # print(train_indices)
    # print(test_indices)
=======
>>>>>>> e3866c46f4ea3e48390a009cf47add22fe43551a

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    training_data_loader = torch.utils.data.DataLoader(dataset,
                                                       sampler=train_sampler,
                                                       batch_size=batchsize,
                                                       num_workers=numworkers,
                                                       pin_memory=pinmemory)
    testing_data_loader = torch.utils.data.DataLoader(dataset,
                                                      sampler=test_sampler,
                                                      batch_size=testBatchsize,
                                                      num_workers=numworkers,
                                                      pin_memory=pinmemory)
    return training_data_loader, testing_data_loader


def TestPredictionPlot(model, device, testing_data_loader, k, d, fout):
    List_t = []
    List_input = []
    List_prediction = []
    List_target = []
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(
                device, torch.float), batch[1].to(device, torch.float)

            prediction = model(input)
            i = 0
            for matrix in input:
<<<<<<< HEAD
                if np.abs(matrix[2][0][0].item() - k) < 1e-8:  # k
                    if np.abs(matrix[3][0][0].item() - d) < 1e-8:  # d
=======
                if matrix[2][0][0].item() == k:  # k
                    if matrix[3][0][0].item() == d:  # d
>>>>>>> e3866c46f4ea3e48390a009cf47add22fe43551a
                        List_t.append(matrix[1][0][0].item())
                        List_input.append(matrix)
                        List_prediction.append(prediction[i])
                        List_target.append(target[i])
                i += 1

    List_input = [x for y, x in sorted(zip(List_t, List_input))]
    List_prediction = [x for y, x in sorted(zip(List_t, List_prediction))]
    List_target = [x for y, x in sorted(zip(List_t, List_target))]
    List_t = sorted(List_t)
    fig, ax = plt.subplots(2, 10, figsize=(20, 3))


    for i in range(10):
        input = List_input[i].cpu().numpy()
        target = List_target[i]
        prediction = List_prediction[i]
        im_tar = ax[0][i].imshow(target[0].cpu().numpy()[::-1], cmap="jet")
        ax[0][i].axis('off')
        ax[0][i].set_title("t = "+str(input[1][0][0]), size=14)

        im_pre = ax[1][i].imshow(prediction[0].cpu().numpy()[::-1], cmap="jet")
        ax[1][i].axis('off')
        fig.subplots_adjust(right=0.8)

    cbar_ax = fig.add_axes([0.82, 0.13, 0.01, 0.75])
    fig.colorbar(im_tar, cax=cbar_ax)
    plt.gcf().text(0.07, 0.6, "K=" +
                str.format('{0:.3f}', input[2][0][0]), fontsize=14)
    plt.gcf().text(0.07, 0.4, "D=" +
                str.format('{0:.3f}', input[3][0][0]), fontsize=14)
    plt.show()
    fig.savefig(fout, dpi=150, quality=100, format='svg')


def TestErrorPlot(model, device, testing_data_loader):
    error_List = []
    testID_List = []
    count = 1

    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(
                device, torch.float), batch[1].to(device, torch.float)

            prediction = model(input)
            tmp_error = 0
            for j in range(len(prediction)):
                # tmp_error = torch.mean(torch.abs(prediction[j]-target[j]))/torch.abs(torch.max(target[j])-torch.min(target[j]))
<<<<<<< HEAD
                # tmp_error = ComputeTestError(prediction[j], target[j]) * 0.0094 / 0.014284110054473812
                tmp_error = ComputeTestError(prediction[j], target[j])
                # if tmp_error < 0.3:
                    # error_List.append(tmp_error.item())
                    # testID_List.append(count)
                    # count += 1
=======
                tmp_error = ComputeTestError(
                    prediction[j], target[j]) * 0.0094 / 0.014284110054473812
>>>>>>> e3866c46f4ea3e48390a009cf47add22fe43551a
                error_List.append(tmp_error.item())
                testID_List.append(count)
                count += 1

    testID_List = np.asarray(testID_List)
    error_List = np.asarray(error_List)
    avg_error = np.average(error_List)
    # print(np.asarray(testID_List).type)
    # print(np.asarray(error_List).size)
    plt.plot(testID_List, error_List, 'ko', zorder=1, markersize=1)
    # plt.scatter(np.asarray(testID_List), np.asarray(error_List), 'bo')
    plt.hlines(avg_error, 1, count, colors='r', zorder=2)
    fig = plt.gcf()
<<<<<<< HEAD
    fig.savefig("./Figure/statistics_testdata.png",
                dpi=300, quality=100, format='png')
=======
    fig.savefig("./Figure/statistics_testdata.svg",
                dpi=300, quality=100, format='svg')
>>>>>>> e3866c46f4ea3e48390a009cf47add22fe43551a
    print(avg_error)


def TestErrorCompute(model, device, testing_data_loader):
    error_List = []
    testID_List = []
    count = 1

    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(
                device, torch.float), batch[1].to(device, torch.float)

            prediction = model(input)
            tmp_error = 0
            for j in range(len(prediction)):
<<<<<<< HEAD
=======
                # tmp_error = torch.mean(torch.abs(prediction[j]-target[j]))/torch.abs(torch.max(target[j])-torch.min(target[j]))
>>>>>>> e3866c46f4ea3e48390a009cf47add22fe43551a
                tmp_error = ComputeTestError(prediction[j], target[j])
                error_List.append(tmp_error.item())
                testID_List.append(count)
                count += 1

    testID_List = np.asarray(testID_List)
    error_List = np.asarray(error_List)
    avg_error = np.average(error_List)

    return avg_error


def ComputeErrorVsEpoch(checkpoint_path, device, testing_data_loader):
    error_list = []
    for i in range(1, 101):
        model = torch.load(checkpoint_path +
                           'model_epoch_' + str(int(i)) + '.pth')
        tmp_error = TestErrorCompute(model, device, testing_data_loader)
        error_list.append(tmp_error)
    error_arr = np.asarray(error_list)
    return error_arr


def ComputeTestError(prediction, target):
    # tmp_error = torch.mean(torch.abs(prediction-target))/torch.abs(torch.max(target)-torch.min(target))
    # tmp_error = torch.mean(torch.abs(prediction-target)**2)/torch.abs(torch.max(target))
    # tmp_error = torch.sqrt(torch.abs(prediction-target).pow(2).sum())/prediction.numel()/torch.abs(torch.max(target)-torch.min(target))
<<<<<<< HEAD
    # tmp_error = torch.sqrt(torch.mean(torch.abs(prediction-target)**2))/torch.abs(torch.max(target))
    tmp_error = torch.sqrt(torch.mean((prediction-target)**2))/torch.abs(torch.max(target)-torch.min(target))
=======
    tmp_error = torch.sqrt(torch.mean(
        torch.abs(prediction-target)**2))/torch.abs(torch.max(target))
>>>>>>> e3866c46f4ea3e48390a009cf47add22fe43551a
    # tmp_error = torch.mean(torch.abs(prediction-target)**2)/torch.abs(torch.max(target)-torch.min(target))
    return tmp_error
