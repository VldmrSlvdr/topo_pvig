import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import random
import numpy as np
from tqdm import tqdm
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch import nn
from read_data import read_mimic
from torch.utils.tensorboard import SummaryWriter  
import pdb
from models.pvig_gaze import pvig_ti_224_gelu
import json
import csv

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    size = len(dataloader.dataset)
    pbar = tqdm(dataloader, total=int(len(dataloader)))
    count = 0
    train_loss = 0.0
    train_acc = 0.0
    for batch, sample in enumerate(pbar):
        x,labels,gaze = sample
        x,labels,gaze = x.to(device), labels.to(device), gaze.to(device)

        outputs = model(x, gaze)
        loss = loss_fn(outputs, labels)
        _,pred = torch.max(outputs,1)
        num_correct = (pred == labels).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        acc = num_correct.item()/len(labels)
        count += len(labels)
        train_loss += loss*len(labels)
        train_acc += num_correct.item()
        pbar.set_description(f"loss: {loss:>f}, acc: {acc:>f}, [{count:>d}/{size:>d}]")
        
    return train_loss/count, train_acc/count
        
def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)

    pbar = tqdm(dataloader, total=int(len(dataloader)))
    count = 0
    test_loss = 0.0
    test_acc = 0.0
    gt = []
    pd = []
    fpr = dict()
    tpr = dict()
    roc_auc = []
    n_classes = 3
    with torch.no_grad():
        for batch, sample in enumerate(pbar):
            x,labels,gaze = sample
            x,labels,gaze = x.to(device), labels.to(device), gaze.to(device)

            outputs = model(x,gaze)
            loss = loss_fn(outputs, labels)
            _,pred = torch.max(outputs,1)
            num_correct = (pred == labels).sum()
            loss = loss.item()
            acc = num_correct.item()/len(labels)
            count += len(labels)
            test_loss += loss*len(labels)
            test_acc += num_correct.item()
            gt.extend(labels.cpu().numpy())
            pd.extend(outputs.cpu().numpy())

            pbar.set_description(f"loss: {loss:>f}, acc: {acc:>f}, [{count:>d}/{size:>d}]")

    gt = np.array(gt); pd = np.array(pd)
    gt = label_binarize(np.array(gt), classes=[0, 1, 2])
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(gt[:, i], pd[:, i])
        roc_auc.append(auc(fpr[i], tpr[i]))
    aucavg = np.mean(roc_auc)
    print("AUC: {}".format(roc_auc))

    return test_loss/count, test_acc/count, aucavg

def save_checkpoint(model, optimizer, epoch, metrics, save_path, is_best=False, best_type=None):
    """Save checkpoint with metrics and training state"""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    torch.save(state, save_path)
    if is_best:
        print(f"Save {best_type}-best model (epoch {epoch}).")


if __name__ == '__main__':

    device =  torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print('Device:', device)
    save_dir = '../output/gazegnn_add3_adam_rotate-test'
    data_dir = '/mnt/f/Datasets/physionet.org/files/mimic_part_jpg'
    writer = SummaryWriter(save_dir)
    batchsize = 32
    n_epochs = 100
    Lr = 1e-4
    evaluate_train = False
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # Create log directory for checkpoints
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save training configuration
    config = {
        'batch_size': batchsize,
        'epochs': n_epochs,
        'learning_rate': Lr,
        'evaluate_train': evaluate_train,
        'data_dir': data_dir,
        'save_dir': save_dir,
        'device': str(device)
    }
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # CSV logging
    csv_file = os.path.join(save_dir, 'training_log.csv')
    with open(csv_file, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'test_auc'])
    
    train_generator,test_generator = read_mimic(batchsize,data_dir)
    
    model = pvig_ti_224_gelu()
    model = model.to(device)
    
    print(model)
    print("There are", sum(p.numel() for p in model.parameters()), "parameters.")
    print("There are", sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable parameters.")
    if torch.cuda.is_available() and torch.cuda.device_count()>1:
      print("Using {} GPUs.".format(torch.cuda.device_count()))
      model = torch.nn.DataParallel(model)
    
    criterion = nn.CrossEntropyLoss().to(device)
    weight_p, bias_p = [],[]
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p +=[p]
        else:
            weight_p +=[p]
    # optimizer = torch.optim.SGD([{'params':weight_p,'weight_decay':1e-4},
    #                              {'params':bias_p,'weight_decay':0}],lr=Lr,momentum=0.9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    idx_best_loss = 0
    idx_best_acc = 0
    idx_best_auc = 0
    
    log_train_loss = []
    log_train_acc = []
    log_test_loss = []
    log_test_acc = []
    log_auc = []
    
    for epoch in range(1, n_epochs+1):
        # print("===> Epoch {}/{}, learning rate: {}".format(epoch, n_epochs, scheduler.get_last_lr()))
        print("===> Epoch {}/{}, learning rate: {}".format(epoch, n_epochs, Lr))
        train_loss, train_acc = train_loop(train_generator, model, criterion, optimizer, device)
        if evaluate_train:
            train_loss, train_acc, test_auc = test_loop(train_generator, model, criterion, device)
        test_loss, test_acc, test_auc = test_loop(test_generator, model, criterion, device)
        print("Training loss: {:f}, acc: {:f}".format(train_loss, train_acc))
        print("Test loss: {:f}, acc: {:f}".format(test_loss, test_acc))
        print("Test AUC: {:.2f}".format(test_auc))
        writer.add_scalar("Trainloss", train_loss, epoch)
        writer.add_scalar("Testloss", test_loss, epoch)
        writer.add_scalar("Trainacc", train_acc, epoch)
        writer.add_scalar("Testacc", test_acc, epoch)
        writer.add_scalar("TestAUC", test_auc, epoch)


        # scheduler.step()
        
        log_train_loss.append(train_loss)
        log_train_acc.append(train_acc)
        log_test_loss.append(test_loss)
        log_test_acc.append(test_acc)
        log_auc.append(test_auc)
        
        # Log to CSV
        with open(csv_file, 'a', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([epoch, train_loss, train_acc, test_loss, test_acc, test_auc])
        
        metrics = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_auc': test_auc
        }
        
        # Save periodic checkpoint (every 10 epochs)
        if epoch % 10 == 0 or epoch == n_epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, metrics, checkpoint_path)
        
        # Save best models (with complete state)
        is_best_loss = test_loss <= log_test_loss[idx_best_loss]
        is_best_acc = test_acc >= log_test_acc[idx_best_acc]
        is_best_auc = test_auc >= log_auc[idx_best_auc]
        
        if is_best_loss:
            save_checkpoint(
                model, optimizer, epoch, metrics,
                os.path.join(save_dir, 'loss_model.pth'), True, 'loss'
            )
            idx_best_loss = epoch - 1
        
        if is_best_acc:
            save_checkpoint(
                model, optimizer, epoch, metrics,
                os.path.join(save_dir, 'acc_model.pth'), True, 'acc'
            )
            idx_best_acc = epoch - 1
        
        if is_best_auc:
            save_checkpoint(
                model, optimizer, epoch, metrics,
                os.path.join(save_dir, 'auc_model.pth'), True, 'auc'
            )
            idx_best_auc = epoch - 1

        print("")

    print("=============================================================")

    print("Loss-best model training loss: {:f}, acc: {:f}".format(log_train_loss[idx_best_loss], log_train_acc[idx_best_loss]))   
    print("Loss-best model test loss: {:f}, acc: {:f}".format(log_test_loss[idx_best_loss], log_test_acc[idx_best_loss]))                
    print("Acc-best model training loss: {:4f}, acc: {:f}".format(log_train_loss[idx_best_acc], log_train_acc[idx_best_acc]))  
    print("Acc-best model test loss: {:f}, acc: {:f}".format(log_test_loss[idx_best_acc], log_test_acc[idx_best_acc]))              
    print("Final model training loss: {:f}, acc: {:f}".format(log_train_loss[-1], log_train_acc[-1]))                 
    print("Final model test loss: {:f}, acc: {:f}".format(log_test_loss[-1], log_test_acc[-1]))           
    
    # Save final metrics summary
    summary = {
        'best_epoch_loss': idx_best_loss + 1,
        'best_loss': log_test_loss[idx_best_loss],
        'best_epoch_acc': idx_best_acc + 1,
        'best_acc': log_test_acc[idx_best_acc],
        'best_epoch_auc': idx_best_auc + 1,
        'best_auc': log_auc[idx_best_auc],
        'final_loss': log_test_loss[-1],
        'final_acc': log_test_acc[-1],
        'final_auc': log_auc[-1]
    }
    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
        
    # Save final model (with complete state for resuming if needed)
    save_checkpoint(
        model, optimizer, n_epochs, 
        {'train_loss': log_train_loss[-1], 'train_acc': log_train_acc[-1],
         'test_loss': log_test_loss[-1], 'test_acc': log_test_acc[-1],
         'test_auc': log_auc[-1]},
        os.path.join(save_dir, 'final_model.pth')
    )
    
    log_train_loss = np.array(log_train_loss)
    log_train_acc = np.array(log_train_acc)
    log_test_loss = np.array(log_test_loss)
    log_test_acc = np.array(log_test_acc)
    log_auc = np.array(log_auc)

    
    plt.figure(figsize=(10, 4))
    plt.subplot(131)
    plt.plot(np.arange(1, n_epochs + 1), log_train_loss)  # train loss (on epoch end)
    plt.plot(np.arange(1, n_epochs + 1), log_test_loss)         #  test loss (on epoch end)
    plt.title("Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.xlim([0, n_epochs])
    plt.legend(['Train', 'Test'], loc="upper left")
    
    plt.subplot(132)
    plt.plot(np.arange(1, n_epochs + 1), log_train_acc)  # train accuracy (on epoch end)
    plt.plot(np.arange(1, n_epochs + 1), log_test_acc)         #  test accuracy (on epoch end)
    plt.title("Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.xlim([0, n_epochs])
    plt.legend(['Train', 'Test'], loc="upper left")    

    plt.subplot(133)
    plt.plot(np.arange(1, n_epochs + 1), log_auc)         #  test accuracy (on epoch end)
    plt.title("AUC")
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.grid()
    plt.xlim([0, n_epochs])
    
    plt.savefig(os.path.join(save_dir, 'log.png'))
    plt.show()
