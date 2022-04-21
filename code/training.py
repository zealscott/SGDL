import numpy as np
import torch
import utils
import dataloader
from utils import timer
import model
import multiprocessing
from sklearn.mixture import GaussianMixture as GMM
from parse import args, log_file
import parse
from scheduler import Scheduler

CORES = multiprocessing.cpu_count() // 2


def memorization_train(dataset, recommend_model, opt):
    Recmodel = recommend_model
    Recmodel.train()

    # sampling
    S = utils.UniformSample(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(parse.device)
    posItems = posItems.to(parse.device)
    negItems = negItems.to(parse.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // args.batch_size + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=args.batch_size)):

        loss, reg_loss = Recmodel.loss(batch_users, batch_pos, batch_neg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        aver_loss += loss.cpu().item()

    aver_loss = aver_loss / total_batch
    timer.zero()
    return f"{aver_loss:.5f}"

def estimate_noise(dataset, recommend_model):
    '''
    estimate noise ratio based on GMM
    '''
    Recmodel: model.LightGCN = recommend_model
    Recmodel.eval()

    dataset: dataloader.MemLoader

    # sampling
    S = utils.UniformSample(dataset)
    users_origin = torch.Tensor(S[:, 0]).long()
    posItems_origin = torch.Tensor(S[:, 1]).long()
    negItems_origin = torch.Tensor(S[:, 2]).long()

    users_origin = users_origin.to(parse.device)
    posItems_origin = posItems_origin.to(parse.device)
    negItems_origin = negItems_origin.to(parse.device)
    with torch.no_grad():
        losses = []
        for (batch_i,
             (batch_users,
              batch_pos,
              batch_neg)) in enumerate(utils.minibatch(users_origin,
                                                       posItems_origin,
                                                       negItems_origin,
                                                       batch_size=args.batch_size)):
            loss, _ = Recmodel.loss(batch_users, batch_pos, batch_neg, reduce=False)
            # concat all losses
            if len(losses) == 0:
                losses = loss
            else:
                losses = torch.cat((losses, loss), dim=0)
        # split losses of each user
        losses_u = []
        st, ed = 0, 0
        for count in dataset.user_pos_counts:
            ed = st + count
            losses_u.append(losses[st:ed])
            st = ed
        # normalize losses of each user
        for i in range(len(losses_u)):
            if len(losses_u[i]) > 1:
                losses_u[i] = (losses_u[i] - losses_u[i].min()) / (losses_u[i].max() - losses_u[i].min())
        losses = torch.cat(losses_u, dim=0)
        losses = losses.reshape(-1, 1).cpu().detach().numpy()
        gmm = GMM(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4)
        gmm.fit(losses)
        prob = gmm.predict_proba(losses)
        prob = prob[:, gmm.means_.argmax()]
        return 1 - np.mean(prob)


def self_guided_train_schedule_reinforce(train_dataset, clean_dataset, recmodel, ltw:model.LTW):
    train_loss, meta_loss = 0, 0
    scheduler = Scheduler(len(recmodel.state_dict())).cuda()
    recmodel.train()
    train_opt = torch.optim.Adam(recmodel.parameters(), lr=args.lr)
    meta_opt = torch.optim.Adam(ltw.parameters(), lr=args.meta_lr)
    schedule_opt = torch.optim.Adam(scheduler.parameters(), lr=args.schedule_lr)

    # sampling
    with timer(name='Train Sample'):
        train_data = utils.UniformSample(train_dataset)
    with timer(name='Clean Sample'):
        clean_data = utils.UniformSample(clean_dataset)

    users = torch.Tensor(train_data[:, 0]).long().to(parse.device)
    posItems = torch.Tensor(train_data[:, 1]).long().to(parse.device)
    negItems = torch.Tensor(train_data[:, 2]).long().to(parse.device)

    users_clean = torch.Tensor(clean_data[:, 0]).long().to(parse.device)
    posItems_clean = torch.Tensor(clean_data[:, 1]).long().to(parse.device)
    negItems_clean = torch.Tensor(clean_data[:, 2]).long().to(parse.device)

    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    users_clean, posItems_clean, negItems_clean = utils.shuffle(users_clean, posItems_clean, negItems_clean)

    total_batch = len(users) // args.batch_size + 1

    clean_data_iter = iter(
        utils.minibatch(users_clean, posItems_clean, negItems_clean, batch_size=args.batch_size))
    for batch_i, (batch_users, batch_pos, batch_neg) in enumerate(utils.minibatch(users,
                                                                                  posItems,
                                                                                  negItems,
                                                                                  batch_size=args.batch_size)):

        try:
            batch_users_clean, batch_pos_clean, batch_neg_clean = next(clean_data_iter)
        except StopIteration:
            clean_data_iter = iter(utils.minibatch(users_clean,
                                                   posItems_clean,
                                                   negItems_clean,
                                                   batch_size=args.batch_size))
            batch_users_clean, batch_pos_clean, batch_neg_clean = next(clean_data_iter)

        weight_for_local_update = list(recmodel.state_dict().values())

        # ============= get input of the scheduler ============= #
        L_theta, _ = recmodel.loss(batch_users_clean, batch_pos_clean, batch_neg_clean, reduce=False)
        L_theta = torch.reshape(L_theta, (len(L_theta), 1))
        v_L_theta = ltw(L_theta.data)

        # assumed update
        L_theta_meta = torch.sum(L_theta * v_L_theta) / len(batch_users_clean)
        recmodel.zero_grad()
        grads = torch.autograd.grad(L_theta_meta, (recmodel.parameters()), create_graph=True, retain_graph=True)

        weight_names = recmodel.weight_names
        for i in range(len(weight_names)):
            recmodel.fast_weights[weight_names[i]] = weight_for_local_update[i] - args.lr * grads[i]
        recmodel.load_state_dict(recmodel.fast_weights)

        L_theta_hat, _ = recmodel.loss(batch_users_clean, batch_pos_clean, batch_neg_clean, reduce=False)

        # for each sample, calculate gradients of 2 losses
        input_embedding_cos = []
        for k in range(len(batch_users_clean)):
            task_grad_cos = []
            grads_theta = torch.autograd.grad(L_theta[k], (recmodel.parameters()), create_graph=True,
                                              retain_graph=True)
            grads_theta_hat = torch.autograd.grad(L_theta_hat[k], (recmodel.parameters()), create_graph=True,
                                                  retain_graph=True)

            # calculate cosine similarity for each parameter
            for j in range(len(grads_theta)):
                task_grad_cos.append(scheduler.cosine(grads_theta[j].flatten().unsqueeze(0),
                                                      grads_theta_hat[j].flatten().unsqueeze(0))[0])
            del grads_theta
            del grads_theta_hat
            # stack similarity of each parameter
            task_grad_cos = torch.stack(task_grad_cos)
            # stack similarity of each sample
            input_embedding_cos.append(task_grad_cos.detach())

        # sample clean data
        weight = scheduler(L_theta, torch.stack(input_embedding_cos).cuda())
        task_prob = torch.softmax(weight.reshape(-1), dim=-1)
        sample_idx = scheduler.sample_task(task_prob, len(batch_users_clean))
        batch_users_clean = batch_users_clean[sample_idx]
        batch_pos_clean = batch_pos_clean[sample_idx]
        batch_neg_clean = batch_neg_clean[sample_idx]

        # ============= training ============= #
        recmodel.load_state_dict(recmodel.keep_weight)

        # assumed update of theta (theta -> theta')
        cost, reg_loss = recmodel.loss(batch_users, batch_pos, batch_neg, reduce=False)
        cost_v = torch.reshape(cost, (len(cost), 1))
        v_lambda = ltw(cost_v.data)

        l_f_meta = torch.sum(cost_v * v_lambda) / len(batch_users)
        recmodel.zero_grad()
        grads = torch.autograd.grad(l_f_meta, (recmodel.parameters()), create_graph=True)

        weight_names = recmodel.weight_names
        for i in range(len(weight_names)):
            recmodel.fast_weights[weight_names[i]] = weight_for_local_update[i] - args.lr * grads[i]

        # load theta' and update params of ltw
        recmodel.load_state_dict(recmodel.fast_weights)

        del grads

        l_g_meta, _ = recmodel.loss(batch_users_clean, batch_pos_clean, batch_neg_clean)

        # REINFORCE
        loss_schedule = 0
        for idx in sample_idx:
            loss_schedule += scheduler.m.log_prob(idx.cuda())
        reward = l_g_meta
        loss_schedule *= reward

        meta_opt.zero_grad()
        l_g_meta.backward(retain_graph=True)
        meta_opt.step()

        schedule_opt.zero_grad()
        loss_schedule.backward()
        schedule_opt.step()

        # reload and actually update theta
        recmodel.load_state_dict(recmodel.keep_weight)
        cost_w, _ = recmodel.loss(batch_users, batch_pos, batch_neg, reduce=False)
        cost_v = torch.reshape(cost_w, (len(cost_w), 1))
        with torch.no_grad():
            w_new = ltw(cost_v)
        loss = torch.sum(cost_v * w_new) / len(batch_users)

        train_opt.zero_grad()
        loss.backward()
        train_opt.step()

        recmodel.store_params()

        train_loss += loss.cpu().item()
        meta_loss += l_g_meta.cpu().item()

    train_loss /= total_batch
    meta_loss /= total_batch
    timer.zero()
    return [f'{train_loss:.5f}', f'{meta_loss:.5f}']


def self_guided_train_schedule_gumbel(train_dataset, clean_dataset, recmodel, ltw:model.LTW):
    train_loss, meta_loss = 0, 0
    scheduler = Scheduler(len(recmodel.state_dict())).cuda()
    recmodel.train()
    train_opt = torch.optim.Adam(recmodel.parameters(), lr=args.lr)
    meta_opt = torch.optim.Adam(ltw.parameters(), lr=args.meta_lr)
    schedule_opt = torch.optim.Adam(scheduler.parameters(), lr=args.schedule_lr)

    # sampling
    train_data = utils.UniformSample(train_dataset)
    clean_data = utils.UniformSample(clean_dataset)

    users = torch.Tensor(train_data[:, 0]).long().to(parse.device)
    posItems = torch.Tensor(train_data[:, 1]).long().to(parse.device)
    negItems = torch.Tensor(train_data[:, 2]).long().to(parse.device)

    users_clean = torch.Tensor(clean_data[:, 0]).long().to(parse.device)
    posItems_clean = torch.Tensor(clean_data[:, 1]).long().to(parse.device)
    negItems_clean = torch.Tensor(clean_data[:, 2]).long().to(parse.device)

    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    users_clean, posItems_clean, negItems_clean = utils.shuffle(users_clean, posItems_clean, negItems_clean)

    total_batch = len(users) // args.batch_size + 1

    clean_data_iter = iter(
        utils.minibatch(users_clean, posItems_clean, negItems_clean, batch_size=args.batch_size))
    for batch_i, (batch_users, batch_pos, batch_neg) in enumerate(utils.minibatch(users,
                                                                                  posItems,
                                                                                  negItems,
                                                                                  batch_size=args.batch_size)):

        try:
            batch_users_clean, batch_pos_clean, batch_neg_clean = next(clean_data_iter)
        except StopIteration:
            clean_data_iter = iter(utils.minibatch(users_clean,
                                                   posItems_clean,
                                                   negItems_clean,
                                                   batch_size=args.batch_size))
            batch_users_clean, batch_pos_clean, batch_neg_clean = next(clean_data_iter)

        weight_for_local_update = list(recmodel.state_dict().values())

        # ============= get input of the scheduler ============= #
        L_theta, _ = recmodel.loss(batch_users_clean, batch_pos_clean, batch_neg_clean, reduce=False)
        L_theta = torch.reshape(L_theta, (len(L_theta), 1))
        v_L_theta = ltw(L_theta.data)

        # assumed update
        L_theta_meta = torch.sum(L_theta * v_L_theta) / len(batch_users_clean)
        recmodel.zero_grad()
        grads = torch.autograd.grad(L_theta_meta, (recmodel.parameters()), create_graph=True, retain_graph=True)

        weight_names = recmodel.weight_names
        for i in range(len(weight_names)):
            recmodel.fast_weights[weight_names[i]] = weight_for_local_update[i] - args.lr * grads[i]
        recmodel.load_state_dict(recmodel.fast_weights)

        L_theta_hat, _ = recmodel.loss(batch_users_clean, batch_pos_clean, batch_neg_clean, reduce=False)

        # for each sample, calculate gradients of 2 losses
        input_embedding_cos = []
        for k in range(len(batch_users_clean)):
            task_grad_cos = []
            grads_theta = torch.autograd.grad(L_theta[k], (recmodel.parameters()), create_graph=True,
                                              retain_graph=True)
            grads_theta_hat = torch.autograd.grad(L_theta_hat[k], (recmodel.parameters()), create_graph=True,
                                                  retain_graph=True)

            # calculate cosine similarity for each parameter
            for j in range(len(grads_theta)):
                task_grad_cos.append(scheduler.cosine(grads_theta[j].flatten().unsqueeze(0),
                                                      grads_theta_hat[j].flatten().unsqueeze(0))[0])
            del grads_theta
            del grads_theta_hat
            # stack similarity of each parameter
            task_grad_cos = torch.stack(task_grad_cos)
            # stack similarity of each sample
            input_embedding_cos.append(task_grad_cos.detach())

        # sample clean data
        weight = scheduler(L_theta, torch.stack(input_embedding_cos).cuda())

        task_prob = torch.softmax(weight.reshape(-1), dim=-1)
        log_p = torch.log(task_prob + 1e-20)
        logits = log_p.repeat([len(log_p), 1])

        sample_idx = scheduler.gumbel_softmax(logits, temperature=args.tau, hard=True)

        if args.model == 'lgn':
            user_emb, pos_emb, neg_emb, _, _, _ = recmodel.getEmbedding(batch_users_clean.long(),
                                                                        batch_pos_clean.long(), batch_neg_clean.long())
        else:
            user_emb, pos_emb, neg_emb = recmodel(batch_users_clean, batch_pos_clean, batch_neg_clean)

        batch_users_clean = torch.mm(sample_idx, user_emb)
        batch_pos_clean = torch.mm(sample_idx, pos_emb)
        batch_neg_clean = torch.mm(sample_idx, neg_emb)

        # ============= training ============= #
        recmodel.load_state_dict(recmodel.keep_weight)

        # assumed update of theta (theta -> theta')
        cost, reg_loss = recmodel.loss(batch_users, batch_pos, batch_neg, reduce=False)
        cost_v = torch.reshape(cost, (len(cost), 1))
        v_lambda = ltw(cost_v.data)

        l_f_meta = torch.sum(cost_v * v_lambda) / len(batch_users)
        recmodel.zero_grad()
        grads = torch.autograd.grad(l_f_meta, (recmodel.parameters()), create_graph=True)

        weight_names = recmodel.weight_names
        for i in range(len(weight_names)):
            recmodel.fast_weights[weight_names[i]] = weight_for_local_update[i] - args.lr * grads[i]

        # load theta' and update params of ltw
        recmodel.load_state_dict(recmodel.fast_weights)

        del grads

        l_g_meta = recmodel.loss_gumbel(batch_users_clean, batch_pos_clean, batch_neg_clean)

        meta_opt.zero_grad()
        l_g_meta.backward(retain_graph=True)
        meta_opt.step()

        schedule_opt.zero_grad()
        l_g_meta.backward()
        schedule_opt.step()

        # reload and actually update theta
        recmodel.load_state_dict(recmodel.keep_weight)
        cost_w, _ = recmodel.loss(batch_users, batch_pos, batch_neg, reduce=False)
        cost_v = torch.reshape(cost_w, (len(cost_w), 1))
        with torch.no_grad():
            w_new = ltw(cost_v)
        loss = torch.sum(cost_v * w_new) / len(batch_users)

        train_opt.zero_grad()
        loss.backward()
        train_opt.step()

        recmodel.store_params()

        train_loss += loss.cpu().item()
        meta_loss += l_g_meta.cpu().item()

    train_loss /= total_batch
    meta_loss /= total_batch
    timer.zero()
    return [f'{train_loss:.5f}', f'{meta_loss:.5f}']

def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in parse.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall),
            'precision':np.array(pre),
            'ndcg':np.array(ndcg)}

def test(dataset, Recmodel, valid=True, multicore=0):
    u_batch_size = args.test_u_batch_size
    dataset: dataloader.Loader
    if valid:
        testDict = dataset.validDict
    else:
        testDict = dataset.testDict

    Recmodel = Recmodel.eval()
    max_K = max(parse.topks)

    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(parse.topks)),
               'recall': np.zeros(len(parse.topks)),
               'ndcg': np.zeros(len(parse.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")

        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            if not valid:
                validDict = dataset.validDict
                for i, user in enumerate(batch_users):
                    try:
                        allPos[i] = np.concatenate((allPos[i], validDict[user]))
                    except KeyError:
                        pass
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(parse.device)
            rating = Recmodel.getUsersRating(batch_users_gpu)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        if multicore == 1:
            pool.close()
        if not valid:
            utils.Logging(log_file, str(results))

    return results

def memorization_test(dataset, Recmodel):
    '''
    memorization procedure,
    update memorization history matrix and generate memorized data
    '''
    u_batch_size = args.test_u_batch_size
    with torch.no_grad():
        users = dataset.trainUniqueUsers
        users_list = []
        items_list = []
        S = utils.sample_K_neg(dataset)
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(parse.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            excluded_users = []
            excluded_items = []
            k_list = []
            for range_i, u in enumerate(batch_users):
                neg_items = S[u]
                items = allPos[range_i]
                k_list.append(len(items))
                neg_items.extend(items)
                excluded_items.extend(neg_items)
                excluded_users.extend([range_i] * (len(neg_items)))

            rating[excluded_users, excluded_items] += 100

            # rating_K: [batch_size, K]
            max_K = max(k_list)
            _, rating_K = torch.topk(rating, k=max_K)
            for i in range(len(rating_K)):
                user = batch_users[i]
                items = rating_K[i].tolist()[:k_list[i]]
                users_list.extend([user] * len(items))
                items_list.extend(items)
            try:
                assert len(users_list) == len(items_list)
            except AssertionError:
                print('len(users_list) != len(items_list)')
            del rating
        dataset.updateMemDict(users_list, items_list)
    return dataset.generate_clean_data()

