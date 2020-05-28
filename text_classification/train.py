import pandas as pd
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel
import sklearn
import logging
import torch
torch.cuda.empty_cache()
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
train=pd.read_csv('train.csv')
train=train[['title','label']]
train=train.rename(columns={'title':'text','label':'labels'})
train.dropna(inplace=True)
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(train['text'],
                                                                            train['labels'],
                                                                            test_size=0.20,
                                                                            random_state=42)
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(train['text'],
                                                                            train['labels'],
                                                                            test_size=0.20,
                                                                            random_state=42)
train_df_clean = pd.concat([X_train_clean, y_train_clean], axis=1)
eval_df_clean = pd.concat([X_test_clean, y_test_clean], axis=1)
train_args = {
    'evaluate_during_training': True,
    'logging_steps': 100,
    'num_train_epochs': 20,
    'evaluate_during_training_steps': 100,
    'save_eval_checkpoints': False,
    'train_batch_size': 32,
    'eval_batch_size': 32,
    'overwrite_output_dir': True,
    'fp16': False,
    'max_seq_length':512
}
model_ALBERT = ClassificationModel('roberta','roberta-base',use_cuda=True,cuda_device=3,args=train_args)
model_ALBERT.train_model(train_df_clean, eval_df=eval_df_clean)
result, model_outputs, wrong_predictions = model_ALBERT.eval_model(eval_df_clean, acc=sklearn.metrics.accuracy_score)
test=pd.read_csv('test.csv')
answer,output=model_ALBERT.predict(test['title'].astype(str))
test['label']=answer
test=test[['id','label']]
test.to_csv('answer.csv',index=False)