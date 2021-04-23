import torch 
import torch.nn as nn
import torch.optim as optim
import dgl
import numpy as np
import pickle as pkl
import argparse
from data_loader import load_data
from TAHIN import TAHIN
from utils import evaluate_auc, evaluate_acc, evaluate_f1_score, evaluate_logloss

def main(args):
    #step 1: Check device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'

    #step 2: Load data 
    g, train_loader, eval_loader, test_loader, meta_paths, user_key, item_key = load_data(args.dataset, args.batch, args.num_workers, args.path)
    g = g.to(device)
    print('Data loaded.')

    #step 3: Create model and training components
    model = TAHIN(
        g, meta_paths, args.in_size, args.out_size, args.num_heads, args.dropout
        )
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    print('Model created.')

    #step 4: Training
    print('Start training.')
    best_acc = 0.0
    kill_cnt = 0
    for epoch in range(args.epochs):
        # Training and validation using a full graph
        model.train()
        train_loss = []
        for step, batch in enumerate(train_loader):
            user, item, label = [_.to(device) for _ in batch]
            logits = model.forward(g, user_key, item_key, user, item)

            # compute loss
            tr_loss = criterion(logits, label)
            train_loss.append(tr_loss)

            # backward
            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()

        train_loss = np.sum(train_loss)

        model.eval()
        with torch.no_grad():
            validate_loss = []
            validate_acc = []
            for step, batch in enumerate(eval_loader):
                user, item, label = [_.to(device) for _ in batch]
                logits = model.forward(g, user_key, item_key, user, item)

                # compute loss
                val_loss = criterion(logits, label)
                val_acc = evaluate_acc(logits.detach().cpu().numpy(), label.detach().cpu().numpy())
                validate_loss.append(val_loss)
                validate_acc.append(val_acc)
        
            validate_loss = np.sum(validate_loss)
            validate_acc = np.mean(validate_acc)
        
            #validate
            if validate_acc > best_acc:
                best_acc = validate_acc
                best_epoch = epoch
                torch.save(model.state_dict(), 'TAHIN'+'_'+args.dataset)
                kill_cnt = 0
                print("saving model...")
            else:
                kill_cnt += 1
                if kill_cnt > args.early_stop:
                    print('early stop.')
                    print("best epoch:{}".format(best_epoch))
                    break

            print("In epoch {}, Train Loss: {:.4f}, Valid Loss: {:.5}\n, Valid ACC: {:.5}".format(epoch, train_loss, validate_loss, validate_acc))

    #test use the best model
    model.eval()
    with torch.no_grad():
        model.load_state_dict(torch.load('TAHIN'+'_'+args.dataset))
        test_loss = []
        test_acc = []
        test_auc = []
        test_f1 = []
        test_logloss = []
        for step, batch in enumerate(test_loader):
            user, item, label = [_.to(device) for _ in batch]
            logits = model.forward(g, user_key, item_key, user, item)

            # compute loss
            loss = criterion(logits, label)
            acc = evaluate_acc(logits.detach().cpu().numpy(), label.detach().cpu().numpy())
            auc = evaluate_auc(logits.detach().cpu().numpy(), label.detach().cpu().numpy())
            f1 = evaluate_f1_score(logits.detach().cpu().numpy(), label.detach().cpu().numpy())
            log_loss = evaluate_logloss(logits.detach().cpu().numpy(), label.detach().cpu().numpy())
            
            test_loss.append(loss)
            test_acc.append(acc)
            test_auc.append(auc)
            test_f1.append(f1)
            test_logloss.append(log_loss)
        
        test_loss = np.sum(test_loss)
        test_acc = np.mean(test_acc)
        test_auc = np.mean(test_auc)
        test_f1 = np.mean(test_f1)
        test_logloss = np.mean(test_logloss)
        print("Test Loss: {:.5}\n, Test ACC: {:.5}\n, AUC: {:.5}\n, F1: {:.5}\n, Logloss: {:.5}\n".format(test_loss, test_acc, test_auc, test_f1, test_logloss))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', default='movielens', help='Dataset to use, default: movielens')
    parser.add_argument('--path', default='./data', help='Path to save the data')
    parser.add_argument('--model', default='TAHIN', help='Model Name')

    parser.add_argument('--batch', default=128, type=int, help='Batch size')
    parser.add_argument('--gpu', type=int, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('--epochs', type=int, default=500, help='Maximum number of epochs')
    parser.add_argument('--wd', type=float, default=0, help='L2 Regularization for Optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of processes to construct batches')
    parser.add_argument('--early_stop', default=15, type=int, help='Patience for early stop.')
    
    parser.add_argument('--in_size', default=128, type=int, help='Initial dimension size for entities.')
    parser.add_argument('--out_size', default=128, type=int, help='Output dimension size for entities.')
    
    parser.add_argument('--num_heads', default=1, type=int, help='Number of attention heads')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout.')
    
    args = parser.parse_args()

    print(args)

    main(args)
    