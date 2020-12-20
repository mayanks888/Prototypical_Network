# coding=utf-8
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser();
    parser.add_argument('-datasets_type', '--dataset_type', type=str, help='type of datasets:omniglot, mini_imagenet, custom_datasets', default='mini_imagenet')
    parser.add_argument('-root', '--dataset_root', type=str, help='path to dataset', default='..' + os.sep + 'dataset')

    parser.add_argument('-exp', '--experiment_root', type=str, help='root where to store models, losses and accuracies', default='..' + os.sep + 'output')

    parser.add_argument('-nep', '--epochs', type=int, help='number of epochs to train for', default=20)

    parser.add_argument('-lr', '--learning_rate', type=float, help='learning rate for the model, default=0.001', default=0.001)

    parser.add_argument('-lrS', '--lr_scheduler_step', type=int, help='StepLR learning rate scheduler step, default=20', default=20)

    parser.add_argument('-lrG', '--lr_scheduler_gamma', type=float, help='StepLR learning rate scheduler gamma, default=0.5', default=0.5)

    parser.add_argument('-its', '--iterations', type=int, help='number of episodes per epoch, default=100', default=100)

    parser.add_argument('-cTr', '--classes_per_it_tr', type=int, help='number of random classes per episode for training, default=60', default=4)

    parser.add_argument('-nsTr', '--num_support_tr', type=int, help='number of samples per class to use as support for training, default=5', default=10)
    parser.add_argument('-nqTr', '--num_query_tr', type=int, help='number of samples per class to use as query for training, default=5', default=10)

    parser.add_argument('-cVa', '--classes_per_it_val', type=int, help='number of random classes per episode for validation, default=5', default=4)

    parser.add_argument('-nsVa', '--num_support_val', type=int, help='number of samples per class to use as support for validation, default=5', default=15)

    parser.add_argument('-nqVa', '--num_query_val', type=int, help='number of samples per class to use as query for validation, default=15', default=15)
    parser.add_argument('-seed', '--manual_seed', type=int, help='input for the manual seeds initializations', default=7)

    parser.add_argument('--cuda', action='store_true', help='enables cuda', default=True)

    # parser.add_argument('--train_image_path', type=str, default='../data/only_traffic_light/training')
    # parser.add_argument('--test_image_path', type=str, default='../data/only_traffic_light/testing')

    parser.add_argument('--train_image_path', type=str, default='/home/mayank_sati/Documents/datsets/traffic_light_gwm/training')
    # parser.add_argument('--train_image_path', type=str, default='/home/mayank_sati/Documents/datsets/traffic_light_gwm/training_all_gwm')
    parser.add_argument('--test_image_path', type=str, default='/home/mayank_sati/Documents/datsets/traffic_light_gwm/testing')

    parser.add_argument('-images_channel', '--img_chanel', type=int, help='channel in input images(rgb or not), default=20',
                        default=3)

    return parser
