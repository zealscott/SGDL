import torch
import time
import training
import model
import pickle
import utils
import dataloader
import parse
from parse import args, log_file
from prettytable import PrettyTable

utils.set_seed(args.seed)
mem_manager = dataloader.MemLoader(args)
train_dataset = dataloader.Loader(args)

Recmodel = model.LightGCN(train_dataset)
Recmodel = Recmodel.to(parse.device)
ltw = model.LTW(args.input, args.hidden1, args.output).cuda()
utils.Logging(log_file, str(args))
results = []
args.lr /= 5
opt = torch.optim.Adam(Recmodel.parameters(), lr=args.lr)

# ========== Phase I: Memorization ========== #
for epoch in range(args.epochs):
    time_train = time.time()
    output_information = training.memorization_train(train_dataset, Recmodel, opt)
    train_log = PrettyTable()
    train_log.field_names = ['Epoch', 'Loss', 'Time', 'Estimated Clean Ratio', 'Memory ratio']

    clean_ratio = training.estimate_noise(mem_manager, Recmodel)
    mem_ratio = training.memorization_test(mem_manager, Recmodel)
    train_log.add_row(
        [f'{epoch + 1}/{args.epochs}', output_information, f'{(time.time() - time_train):.3f}',
         f'{clean_ratio:.5f}', f'{mem_ratio:.5f}']
    )
    utils.Logging(log_file, str(train_log))

    # memorization point
    if mem_ratio >= clean_ratio:
        utils.Logging(log_file, f'==================Memorization Point==================')
        break

trans_epoch = epoch
clean_dataset = dataloader.CleanLoader(args)
args.lr *= 5
best_epoch = epoch

# ========== Phase II: Self-Guided Learning ========== #
for epoch in range(trans_epoch, args.epochs):
    if epoch % args.eval_freq == 0:
        utils.Logging(log_file, f'======================Validation======================')
        valid_log = PrettyTable()
        valid_log.field_names = ['Precision', 'Recall', 'NDCG', 'Current Best Epoch']
        valid_result = training.test(train_dataset, Recmodel, valid=True, multicore=args.multicore)
        results.append(valid_result)
        with open('./{}/results_{}_{}.pkl'.format(
                args.dataset,
                args.lr,
                args.meta_lr
        ), 'wb') as f:
            pickle.dump(results, f)
        is_stop, is_save = utils.EarlyStop(results)

        # save current best model
        if is_save:
            best_epoch = epoch
            torch.save(Recmodel.state_dict(), './{}/model_{}_{}.pth'.format(
                args.dataset,
                args.lr,
                args.model
            ))
        valid_log.add_row(
            [valid_result['precision'][0], valid_result['recall'][0], valid_result['ndcg'][0], best_epoch]
        )
        utils.Logging(log_file, str(valid_log))
        if is_stop:
            break

    time_train = time.time()
    if args.schedule_type == 'reinforce':
        output_information = training.self_guided_train_schedule_reinforce(train_dataset, clean_dataset, Recmodel, ltw)
    elif args.schedule_type == 'gumbel':
        output_information = training.self_guided_train_schedule_gumbel(train_dataset, clean_dataset, Recmodel, ltw)
    else:
        utils.Logging(log_file, 'Invalid scheduler type !')
        exit()

    train_log = PrettyTable()
    train_log.field_names = ['Epoch', 'Train Loss', "Meta Loss", "Time"]
    train_log.add_row(
        [f'{epoch + 1}/{args.epochs}', output_information[0], output_information[1], f'{(time.time()-time_train):.3f}']
    )
    utils.Logging(log_file, str(train_log))

# ========== Test ========== #
utils.Logging(log_file, f'=========================Test=========================')
state = torch.load('./{}/model_{}_{}.pth'.format(
                args.dataset,
                args.lr,
                args.model
            ))
Recmodel.load_state_dict(state)
training.test(train_dataset, Recmodel, valid=False, multicore=args.multicore)
