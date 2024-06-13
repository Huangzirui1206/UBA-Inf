import torch
import torch.nn.functional as F

from utils.trainer_cls import given_dataloader_test, all_acc

class aggregator(): # input several sub-models and perform like a entire model
    def __init__(self,
                 models,
                 weight='uniform') -> None:
        self.models = models
        if type(weight) == list:
            assert len(self.models) == len(weight)
            self.weight = weight
        elif weight == 'uniform':
            self.weight = [1 for _ in range(len(self.models))]
            
    # To make the aggregator act like a single model
    def __call__(self, img):
        votes = [F.normalize(F.softmax(model(img),dim=1), p=1, dim=1) for model in self.models]
        proba = None
        for vote in votes:
            if proba is None:
                proba = vote
            else:
                proba = proba + vote
        proba = proba / len(votes)
        return proba

    def eval(self):
        for model in self.models:
            model.eval()
        return self
    
    def train(self): # the sub-models should be pre-trained
        raise ValueError('An aggregator can\'t be trained.')
    
    def to(self, device, non_blocking=True):
        for model in self.models:
            model.to(device, non_blocking=non_blocking)
    
    # for model test, reference to utils.trainer_cls.PureCleanModelTrainer        
    def test_given_dataloader(self, test_dataloader, device = None, verbose = 0, non_blocking=True):

        if device is None:
            device = self.device

        model = self
        non_blocking = non_blocking

        return given_dataloader_test(
                    model,
                    test_dataloader,
                    self.criterion,
                    non_blocking,
                    device,
                    verbose,
            )[0]
        
    def test_given_dataloader_on_mix(self, test_dataloader, device = None, verbose = 0):

        if device is None:
            device = self.device

        model = self
        model.to(device, non_blocking=self.non_blocking)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss_sum_over_batch': 0,
            'test_total': 0,
        }

        criterion = self.criterion.to(device, non_blocking=self.non_blocking)

        if verbose == 1:
            batch_predict_list = []
            batch_label_list = []
            batch_original_index_list = []
            batch_poison_indicator_list = []
            batch_original_targets_list = []

        with torch.no_grad():
            for batch_idx, (x, labels, original_index, poison_indicator, original_targets, *other_info) in enumerate(test_dataloader):
                x = x.to(device, non_blocking=self.non_blocking)
                labels = labels.to(device, non_blocking=self.non_blocking)
                pred = model(x)
                loss = criterion(pred, labels.long())

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(labels).sum()

                if verbose == 1:
                    batch_predict_list.append(predicted.detach().clone().cpu())
                    batch_label_list.append(labels.detach().clone().cpu())
                    batch_original_index_list.append(original_index.detach().clone().cpu())
                    batch_poison_indicator_list.append(poison_indicator.detach().clone().cpu())
                    batch_original_targets_list.append(original_targets.detach().clone().cpu())

                metrics['test_correct'] += correct.item()
                metrics['test_loss_sum_over_batch'] += loss.item()
                metrics['test_total'] += labels.size(0)

        metrics['test_loss_avg_over_batch'] = metrics['test_loss_sum_over_batch']/len(test_dataloader)
        metrics['test_acc'] = metrics['test_correct'] / metrics['test_total']

        if verbose == 0:
            return metrics, \
                   None, None, None, None, None
        elif verbose == 1:
            return metrics, \
                   torch.cat(batch_predict_list), \
                   torch.cat(batch_label_list), \
                   torch.cat(batch_original_index_list), \
                   torch.cat(batch_poison_indicator_list), \
                   torch.cat(batch_original_targets_list)
    
    def backdoorTest(self, 
                       clean_test_dataloader, 
                       bd_test_dataloader, 
                       device,
                       criterion,
                       non_blocking = False):
        
        self.device = device
        self.non_blocking = non_blocking
        self.criterion = criterion
        
        clean_metrics = self.test_given_dataloader(clean_test_dataloader, verbose=1)

        clean_test_loss_avg_over_batch = clean_metrics["test_loss_avg_over_batch"]
        test_acc = clean_metrics["test_acc"]

        bd_metrics, \
        bd_test_epoch_predict_list, \
        bd_test_epoch_label_list, \
        bd_test_epoch_original_index_list, \
        bd_test_epoch_poison_indicator_list, \
        bd_test_epoch_original_targets_list = self.test_given_dataloader_on_mix(bd_test_dataloader, verbose=1)

        bd_test_loss_avg_over_batch = bd_metrics["test_loss_avg_over_batch"]
        test_asr = all_acc(bd_test_epoch_predict_list, bd_test_epoch_label_list)
        test_ra = all_acc(bd_test_epoch_predict_list, bd_test_epoch_original_targets_list) 
        
        return {
            "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
            "bd_test_loss_avg_over_batch" : bd_test_loss_avg_over_batch,
            "test_acc" : test_acc,
            "test_asr" : test_asr,
            "test_ra" : test_ra,
        }  
        
         