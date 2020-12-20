
from prototypical_batch_sampler import PrototypicalBatchSampler
from prototypical_loss import prototypical_loss as loss_fn
from omniglot_dataset import OmniglotDataset
from protonet import ProtoNet
from parser_util import get_parser
import torchvision.datasets as dset

from tqdm import tqdm
import numpy as np
import torch
import os
import argparse
from custom_datsets_load import datasets_custom,MiniImageNet
import torch.nn.functional as F
from samplers import CategoriesSampler
from torch.utils.data import DataLoader

def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


def init_dataset(opt, mode):
    dataset = OmniglotDataset(mode=mode, root=opt.dataset_root)
    n_classes = len(np.unique(dataset.y))
    if n_classes < opt.classes_per_it_tr or n_classes < opt.classes_per_it_val:
        raise(Exception('There are not enough classes in the dataset in order ' +
                        'to satisfy the chosen classes_per_it. Decrease the ' +
                        'classes_per_it_{tr/val} option and try again.'))
    return dataset


def init_sampler(opt, labels, mode):
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=opt.iterations)


def init_dataloader(opt, mode):
    dataset = init_dataset(opt, mode)
    sampler = init_sampler(opt, dataset.y, mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return dataloader


def init_protonet(opt):
    '''
    Initialize the ProtoNet
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    model = ProtoNet().to(device)
    return model


def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate)


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):
    '''
    Train the model with the prototypical learning algorithm
    '''

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')

    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        model.train()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=opt.num_support_tr)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
        avg_loss = np.mean(train_loss[-opt.iterations:])
        avg_acc = np.mean(train_acc[-opt.iterations:])
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        lr_scheduler.step()
        if val_dataloader is None:
            continue
        val_iter = iter(val_dataloader)
        model.eval()
        for batch in val_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=opt.num_support_val)
            val_loss.append(loss.item())
            val_acc.append(acc.item())
        avg_loss = np.mean(val_loss[-opt.iterations:])
        avg_acc = np.mean(val_acc[-opt.iterations:])
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
            avg_loss, avg_acc, postfix))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()

    # torch.save(model.state_dict(), last_model_path)

    def save_model(name):
        torch.save(model.state_dict(), last_model_path)

    save_model('epoch-{}'.format(epoch))


    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        save_list_to_file(os.path.join(opt.experiment_root,
                                       name + '.txt'), locals()[name])

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


def test(opt, test_dataloader, model):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    avg_acc = list()
    for epoch in range(10):
        test_iter = iter(test_dataloader)
        for batch in test_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            _, acc = loss_fn(model_output, target=y,
                             n_support=opt.num_support_val)
            avg_acc.append(acc.item())
    avg_acc = np.mean(avg_acc)
    print('Test Acc: {}'.format(avg_acc))

    return avg_acc


def eval(opt):
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    test_dataloader = init_dataset(options)[-1]
    model = init_protonet(options)
    model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))

    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)


def main():


    # options= parser.parse_args()

    options = get_parser().parse_args()
    print(vars(options))
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # set_gpu(options.gpu)
    # ensure_path(options.save_path)

    # trainset = MiniImageNet('train')
    # trainset = MiniImageNet('train')
##################################################333
    if options.dataset_type=="custom_datasets":
    #loading datasets
        # train_folder_dataset = dset.ImageFolder(root=options.train_image_path)
        # trainset = datasets_custom("train", train_folder_dataset)
        # train_sampler = CategoriesSampler(trainset.label, options.iterations, options.classes_per_it_tr, options.num_support_tr + options.num_query_tr)
        # train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=4, pin_memory=True)
        # tr_dataloader=train_loader

        # valset = MiniImageNet('val')
        valset_folder_dataset = dset.ImageFolder(root=options.test_image_path)
        valset = datasets_custom("test", valset_folder_dataset)
        val_sampler = CategoriesSampler(valset.label, 400, options.classes_per_it_val, options.num_support_val + options.num_query_val)
        val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)
        val_dataloader=val_loader

        options.img_chanel=3
    elif options.dataset_type=="mini_imagenet":
        # trainset = MiniImageNet('train')
        # train_sampler = CategoriesSampler(trainset.label, options.iterations, options.classes_per_it_tr, options.num_support_tr + options.num_query_tr)
        # tr_dataloader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=4, pin_memory=True)

        valset = MiniImageNet('val')
        val_sampler = CategoriesSampler(valset.label, 400, options.classes_per_it_val, options.num_support_val + options.num_query_val)
        val_dataloader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)
        options.img_chanel = 3
    elif options.dataset_type=="omniglot":
        # tr_dataloader = init_dataloader(options, 'train')
        val_dataloader = init_dataloader(options, 'val')
        # trainval_dataloader = init_dataloader(options, 'trainval')
        # test_dataloader = init_dataloader(options, 'test')
        options.img_chanel = 1
    else:
        print("Wrong input for datasets name")

    #########################################################################################
    init_seed(options)
#########################################################33
    # tr_dataloader = init_dataloader(options, 'train')
    # # tr_dataloader = init_dataloader(options, 'val')
    # val_dataloader=tr_dataloader
    # test_dataloader=tr_dataloader
    # # val_dataloader = init_dataloader(options, 'val')
    # # trainval_dataloader = init_dataloader(options, 'trainval')
    # # test_dataloader = init_dataloader(options, 'test')
######################################################3
    model = init_protonet(options)
    print('Testing with last model..')
    # test(opt=options, test_dataloader=test_dataloader, model=model)
    #
    model.load_state_dict(torch.load('../output/best_model.pth'))
    # print('Testing with best model..')
    acc = test(opt=options, test_dataloader=val_dataloader, model=model)
    print(acc)


if __name__ == '__main__':
    main()
