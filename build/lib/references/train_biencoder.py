# based off of ben litterer's code
from tqdm.auto import tqdm
import torch 
import transformers
from transformers import PreTrainedTokenizer
from transformers import RobertaTokenizer, PreTrainedTokenizer, DistilBertTokenizer, DistilBertModel, RobertaModel
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses, util
from datasets import Dataset
import pandas as pd
from rapidfuzz import fuzz, process
from transformers.optimization import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt 
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from torch.nn import CosineEmbeddingLoss
import random
from torch.nn import CosineEmbeddingLoss
from torch import nn
import time

import os
import re
import argparse
from rapidfuzz import fuzz, process
from data.bible_utils import comp_bible_helper

class BiModel(nn.Module): 
    def __init__(self, directory='sentence-transformers/all-mpnet-base-v2', device=0):
        super(BiModel,self).__init__()
        self.model = AutoModel.from_pretrained(directory)
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-4)
        
    def mean_pooling(self, token_embeddings, attention_mask): 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask): 
        input_ids_a = input_ids[0].to(device)
        input_ids_b = input_ids[1].to(device)
        attention_a = attention_mask[0].to(device)
        attention_b = attention_mask[1].to(device)
        
        #encode sentence and get mean pooled sentence representation 
        encoding1 = self.model(input_ids_a, attention_mask=attention_a)[0] # accesses token embeddings
        encoding2 = self.model(input_ids_b, attention_mask=attention_b)[0]
        
        meanPooled1 = self.mean_pooling(encoding1, attention_a)
        meanPooled2 = self.mean_pooling(encoding2, attention_b)
        
        pred = self.cos(meanPooled1, meanPooled2)
        return pred

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_out_dir', default="/data/laviniad/sermons-ir/modeling/tuned_mpnet/model.pth", type=str)
    parser.add_argument('--congress_errata_path', default="/data/laviniad/congress_errata/", type=str)
    parser.add_argument('--batchsize', default=32, type=int)
    parser.add_argument('--max_neg_to_pos', default=3, type=int)
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument('--k_folds', default=3, type=int)
    parser.add_argument('--subset_positive', action='store_true', help="This tells the script to only take the fuzzily matched examples and none of the other quotes. If this option is not passed, the script will instead choose the optimal quote for *every* quote in the df, from the popular verse list.") # number of examples 
    parser.add_argument('--new_fsm_threshold', default=-1, type=int)
    parser.add_argument('--debug', action='store_true')
    #parser.add_argument('--keyword_path', default='/data/laviniad/sermons-ir/mask-predictions/50words_5min.json', type=str)
    parser.add_argument('--device', default="2", type=str)
    args = parser.parse_args()

    #os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

    # load data
    LABELS = {'negative': 0.0, 'positive_quote': 0.9, 'positive_cite': 0.7, 'parallel': 1.0}
    negative_examples_raw = pd.read_csv(args.congress_errata_path + 'negative_examples_df.csv')
    positive_examples_raw = pd.read_csv(args.congress_errata_path + 'positive_examples_df.csv')
    parallel_examples_raw = pd.read_csv(args.congress_errata_path + 'parallel_positive_examples_df.csv')

    TAG_RE = re.compile(r'<[^>]+>')

    def remove_tags(text):
        return TAG_RE.sub('', text)

    bible_df = comp_bible_helper()
    pop_verses = pd.read_csv('/home/laviniad/projects/religion_in_congress/data/most_popular_verses.csv')
    n = 1000 # VERY generous
    # remove the first verse, which is 'UNKNOWN' (artifact of sermon data)
    pop_citations = list(pop_verses['verse'].iloc[1:n+2])

    bible_df['King James Bible'] = bible_df['King James Bible'].apply(remove_tags)
    bible_df['Verse'] = bible_df['Verse'].apply(lambda x: x.lower())
    limited_bible_df = bible_df[bible_df['Verse'].apply(lambda x: x in pop_citations)]
    limited_verses = limited_bible_df['King James Bible']
    limited_verse_to_citation = dict(zip(limited_verses, limited_bible_df['Verse']))
    limited_citation_to_verse = {v: k for k,v in limited_verse_to_citation.items()}

    print("Labeling and loading data")
    example_df = []

    if args.subset_positive:
        for idx,row in positive_examples_raw.iterrows():
            if row['verse_quoted'] != 'no_quote':
                example_df.append({'text_one': row['text'],
                                   'text_two': row['verse_text'],
                                   'label': LABELS['positive_quote']
                                  })
            elif row['regex_citation'] != 'no_citation':
                citation = row['regex_citation']
                if citation.lower() in limited_citation_to_verse.keys():
                    example_df.append({'text_one': row['text'],
                                       'text_two': limited_citation_to_verse[citation.lower()],
                                       'label': LABELS['positive_cite']
                                      })
    elif args.new_fsm_threshold > 0:
        for idx,row in positive_examples_raw.iterrows():
            result = process.extractOne(row['text'], limited_verses, scorer=fuzz.token_sort_ratio)
            verse, score = result[0], result[1]
            if score >= args.new_fsm_threshold:
                example_df.append({'text_one': row['text'],
                                    'text_two': verse,
                                    'label': LABELS['positive_quote']
                                    })
    else:
        for idx,row in positive_examples_raw.iterrows():
            result = process.extractOne(row['text'], limited_verses, scorer=fuzz.token_sort_ratio)
            verse, score = result[0], result[1]
            example_df.append({'text_one': row['text'],
                                'text_two': verse,
                                'label': LABELS['positive_quote']
                                })

    num_pairs_from_each_verse = 2
    for idx,row in parallel_examples_raw.iterrows():
        verse_versions = row['verse_versions']
        sampled_pairs = [(random.sample(verse_versions, 2)) for _ in range(num_pairs_from_each_verse)]

        for random_two in sampled_pairs:
            example_df.append({'text_one': random_two[0],
                               'text_two': random_two[1],
                               'label': LABELS['parallel']
                              })

    sample = negative_examples_raw.sample(int(len(example_df) * args.max_neg_to_pos))
    for idx,row in sample.iterrows():
        example_df.append({'text_one': row['text'],
                           'text_two': row['random_verse'],
                           'label': LABELS['negative']
                          })

    example_df = pd.DataFrame(example_df)
    print(f"Counts of each label in the dataframe: {example_df['label'].value_counts()}")

    # set hyperparams
    EPOCHS = args.num_epochs
    FOLDS = args.k_folds
    SEED = 50
    BATCH_SIZE = args.batchsize

    # set seeds 
    torch.manual_seed(SEED)
    random.seed(SEED)


    def validateBi(model, validLoader, loss_func):
        model.eval()

        preds = []
        gts = []

        for batch in tqdm(validLoader):
            input_ids = [batch["text_one_input_ids"], batch["text_two_input_ids"]]
            attention_masks = [batch["text_one_attention_mask"], batch["text_two_attention_mask"]]
            pred = model(input_ids, attention_masks)
            gt = batch["label"].to(device)

            preds += list(pred.detach().cpu().tolist())
            gts += list(gt.detach().cpu().tolist())

        corr = np.corrcoef(preds, gts)[1,0]
        print(f"Correlation: {corr}")
        model.train()
        return corr


    def trainBi(trainDataset, validDataset): 
        model = BiModel().to(device)
        optim = torch.optim.Adam(model.parameters(), lr=2e-6)

        total_steps = int(len(trainDataset) / BATCH_SIZE) * EPOCHS
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps - warmup_steps)

        loss_func = torch.nn.MSELoss(reduction="mean")

        trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
        validLoader = torch.utils.data.DataLoader(validDataset, batch_size=4, shuffle=False)

        corrList = []
        for epoch in range(EPOCHS):
            print("EPOCH: " + str(epoch))
            corrList.append(validateBi(model, validLoader, loss_func))

            model.train()  # make sure model is in training mode

            for batch in tqdm(trainLoader):
                optim.zero_grad()

                input_ids = [batch["text_one_input_ids"], batch["text_one_input_ids"]]
                attention_masks = [batch["text_one_attention_mask"], batch["text_two_attention_mask"]]
                pred = model(input_ids, attention_masks)

                gt = batch["label"].to(device)
                loss = loss_func(pred, gt)

                # using loss, calculate gradients and then optimize
                loss.backward()
                optim.step()
                scheduler.step()

        print("final validation")
        corrList.append(validateBi(model, validLoader, loss_func))
        return corrList, model


    from sklearn.model_selection import KFold
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    metrics = []
    transformers.logging.set_verbosity_error()
    biTokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')

    biCorrs = []
    model_objs = []
    st = time.time()

    #we create splits based on the position (not the actual index) of rows
    for i, (train_index, valid_index) in enumerate(kf.split(example_df)): 

        #grab the rows in enDf corresponding to the positions of our split 
        validDf = example_df.iloc[valid_index]

        #now get the actual indices that have been selected
        #and subtract the indices in trainDf away from those 
        remainingIndices = list(set(example_df.index) - set(validDf.index))
        trainDf = example_df.loc[remainingIndices]
        validDf = validDf.reset_index(drop=True)

        print("###### " + str(i).upper() + " ######")
        print("Train df len: " + str(len(trainDf)))
        print("Valid df len: " + str(len(validDf)))


        trainDataset = Dataset.from_pandas(trainDf)
        validDataset = Dataset.from_pandas(validDf)

        all_cols = ["label"]
        for part in ["text_one", "text_two"]:
            trainDataset = trainDataset.map(lambda x: biTokenizer(x[part], max_length=512, padding="max_length", truncation=True))
            validDataset = validDataset.map(lambda x: biTokenizer(x[part], max_length=512, padding="max_length", truncation=True))

            for col in ['input_ids', 'attention_mask']: 
                trainDataset = trainDataset.rename_column(col, part+'_'+col)
                validDataset = validDataset.rename_column(col, part+'_'+col)
                all_cols.append(part+'_'+col)

        trainDataset.set_format(type='torch', columns=all_cols)
        validDataset.set_format(type='torch', columns=all_cols)
        corrs, model = trainBi(trainDataset, validDataset)

        biCorrs.append(corrs)
        model_objs.append(model)

    # save best model
    best_model = np.argmax(np.asarray(biCorrs)[:,0])
    torch.save(model_objs[best_model].state_dict(), args.model_out_dir)

    et = time.time()
    elapsed = et - st

    import pickle
    RESULTS_PATH = "/data/laviniad/sermons-ir/modeling/mpnet_results"

    with open(RESULTS_PATH + "/data.pkl", "wb") as f:
        pickle.dump(biCorrs, f)

    with open(RESULTS_PATH + "/elapsed_time.pkl", "wb") as f:
        pickle.dump(elapsed, f)
                
