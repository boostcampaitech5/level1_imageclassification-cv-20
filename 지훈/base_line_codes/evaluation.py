import torch 
from sklearn.metrics import f1_score, accuracy_score

class Evaluation():

    def __init__(self,num_classes : int) -> None:

        self.num_classes = num_classes
        self.preds = []
        self.labels = []
        self.correct_num = [ 0 for _ in range(num_classes)]
        self.total_num = [ 0 for _ in range(num_classes)]

        
    def update(self,pred : torch.tensor, label : torch.tensor):
        # pred : (batchsize,num_class)
        # label : (batchsize)
        # pred argmax로 변환하기 
        pred = torch.argmax(pred,dim = 1)
        self.preds.extend(pred.tolist())
        self.labels.extend(label.tolist())

        for i in range(len(pred)):
            answer = label[i]
            self.total_num[answer] += 1
            if pred[i] == answer:
                self.correct_num[answer] += 1
                
        return (pred == label).sum().item() / len(pred)     
        
        
            
    def result(self , option = 'all',f1_score_option = 'macro'):
        # total accuracy, total f1_score 출력
        if option == 'all':
            # 라벨별 정확도, 총 정확도, F1_score
            label_accs = []
            for i in range(self.num_classes):
                if self.total_num[i]:
                    label_accs.append(round(self.correct_num[i] / self.total_num[i], 3))
                else:
                    label_accs.append(0)
            return label_accs, round(accuracy_score(self.preds, self.labels), 3), round(f1_score(self.preds, self.labels, average = f1_score_option), 3)      

