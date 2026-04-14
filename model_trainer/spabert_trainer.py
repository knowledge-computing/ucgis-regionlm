
import logging
import os, sys
import glob
import re
from tqdm import tqdm

import pandas as pd
import geopandas as gpd

from shapely import Point, wkt

import torch
from torch.utils.data import DataLoader

from transformers import AdamW, BertTokenizer
from transformers.models.bert.modeling_bert import BertForMaskedLM

from spabert.models.spatial_bert_model import SpatialBertConfig, SpatialBertForMaskedLM, SpatialBertModel
from spabert.utils.common_utils import load_spatial_bert_pretrained_weights, get_spatialbert_embedding

from model_trainer._base import ModelTrainerBase
from model_trainer._utils.pseudo_sentence_loader import PseudoSentenceLoader
from utils import const


class SpaBERTTrainer(ModelTrainerBase):
    def __init__(self):
        self.max_token_len = 512
        self.distance_norm_factor = 0.0001
        self.spatial_dist_fill = 100

    def train_model(self,
                    json_file_path,
                    model_save_dir,
                    num_workers=5,
                    shuffle_training_data=True,
                    batch_size=12,
                    epochs=10,
                    lr = 5e-5,
                    save_interval=2000,
                    with_type = False,
                    sep_between_neighbors = False,
                    freeze_backbone= False,
                    bert_option='bert-base',
                    no_spatial_distance = False,
                    verbose=False):       
        try:
            self.json_file_path = json_file_path
            self.num_workers = num_workers
            self.shuffle_training_data = shuffle_training_data
            self.batch_size = batch_size
            self.epochs = epochs
            self.lr = lr
            self.save_interval = save_interval
            self.with_type = with_type
            self.sep_between_neighbors = sep_between_neighbors
            self.freeze_backbone = freeze_backbone
            self.bert_option = bert_option
            self.no_spatial_distance = no_spatial_distance
            self.model_save_dir = model_save_dir
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            self.verbose = verbose

            if self.verbose:
                logging.info('SpaBERT model saved to ' + os.path.abspath(self.model_save_dir))
            
            #Setup BERT
            if bert_option == 'bert-base':
                bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                config = SpatialBertConfig(use_spatial_distance_embedding = not self.no_spatial_distance)
            elif bert_option == 'bert-large':
                bert_model = BertForMaskedLM.from_pretrained('bert-large-uncased')
                self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
                config = SpatialBertConfig(use_spatial_distance_embedding = not self.no_spatial_distance, hidden_size = 1024, intermediate_size = 4096, num_attention_heads=16, num_hidden_layers=24)
            else:
                raise NotImplementedError

            self.model = SpatialBertForMaskedLM(config)
    
            #model_weights = sorted(glob.glob(os.path.join(model_save_dir, "*.pth")), key=os.path.getmtime)

            #if len(model_weights) == 0:
            self.model.load_state_dict(bert_model.state_dict(), strict = False) # load sentence position embedding weights as well
            print("BERT pretrained weights loaded")  
            start_epoch = 0
            # else:
            #     self.model.load_state_dict(torch.load(model_weights[-1]))
            #     start_epoch = int(re.search('_ep(.*)_iter', model_weights[-1]).group(1))
            #     print("SpaBERT pretrained weights", model_weights[-1].split("/")[-1], ' loaded')
    
            if bert_option == 'bert-large' and freeze_backbone:
                print('freezing backbone weights')
                for param in self.model.parameters():
                    param.requires_grad = False
                for param in  self.model.cls.parameters(): 
                    param.requires_grad = True
                for param in  self.model.bert.encoder.layer[21].parameters(): 
                    param.requires_grad = True
                for param in  self.model.bert.encoder.layer[22].parameters(): 
                    param.requires_grad = True
                for param in  self.model.bert.encoder.layer[23].parameters(): 
                    param.requires_grad = True
        
            train_dataset = PseudoSentenceLoader(
                data_file_path = self.json_file_path,
                tokenizer = self.tokenizer,
                max_token_len = self.max_token_len, 
                distance_norm_factor = self.distance_norm_factor, 
                spatial_dist_fill=self.spatial_dist_fill
            )
        
            train_loader = DataLoader(train_dataset, batch_size= self.batch_size, num_workers=self.num_workers,
                                        shuffle=self.shuffle_training_data, pin_memory=True, drop_last=True)

            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            self.model.to(device)
            self.model.train()

            # initialize optimizer
            optim = AdamW(self.model.parameters(), lr = self.lr)

            print('start training...')

            for epoch in range(start_epoch, self.epochs):
                # setup loop with TQDM and dataloader
                loop = tqdm(train_loader, leave=True)
                iter = 0
                for batch in loop:
                    # initialize calculated gradients (from prev step)
                    optim.zero_grad()
                    # pull all tensor batches required for training
                    input_ids = batch['masked_input'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    position_list_x = batch['norm_lng_list'].to(device)
                    position_list_y = batch['norm_lat_list'].to(device)
                    sent_position_ids = batch['sent_position_ids'].to(device)
                    labels = batch['pseudo_sentence'].to(device)

                    outputs = self.model(input_ids, attention_mask = attention_mask, sent_position_ids = sent_position_ids,
                        position_list_x = position_list_x, position_list_y = position_list_y, labels = labels)

                    loss = outputs.loss
                    loss.backward()
                    optim.step()

                    loop.set_description(f'Epoch {epoch}')
                    loop.set_postfix({'loss':loss.item()})

                    if self.verbose:
                        print('ep'+str(epoch)+'_' + '_iter'+ str(iter).zfill(5), loss.item() )

                    iter += 1

                    if iter % save_interval == 0 or iter == loop.total:
                        
                        save_path = os.path.join(model_save_dir, str(epoch) + '_iter'+ str(iter).zfill(5) \
                        + '_' +str("{:.4f}".format(loss.item())) +'.pth' )
                        torch.save(self.model.state_dict(), save_path)
                        print('saving model checkpoint to', save_path)
                if self.verbose:
                    logging.info("SpaBERT model trained successfully.")
        except Exception as e:
            logging.error(f"Error training Random Forest model: {e}")

    def predict(self, json_file_path, model_save_dir, csv_file_path, verbose=False):
        try:
            self.json_file_path = json_file_path
            self.model_weight_path = model_save_dir
            self.csv_file_path = csv_file_path
            self.verbose = verbose

            config = SpatialBertConfig()
            self.model = SpatialBertModel(config)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            self.model.eval()

            model_weights = sorted(glob.glob(os.path.join(self.model_weight_path, "*.pth")), key=os.path.getmtime)
            self.model = load_spatial_bert_pretrained_weights(self.model, model_weights[-1])
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

            json_dataset = PseudoSentenceLoader(
                data_file_path=self.json_file_path,
                tokenizer=tokenizer,
                max_token_len=self.max_token_len,
                distance_norm_factor=self.distance_norm_factor,
                spatial_dist_fill=self.spatial_dist_fill
            )

            entity_ids = []
            entity_context = []
            entity_coords = []
            entity_emb = []

            #the following column names follow the SpaBERT convention
            for entity in tqdm(json_dataset):
                spabert_emb = get_spatialbert_embedding(entity, self.model)
                entity_ids.append(entity['pivot_id'])
                entity_context.append(entity['pivot_name'])
                entity_coords.append(Point(entity['pivot_pos'][::-1]))
                entity_emb.append(spabert_emb.tolist())
                
            df = pd.DataFrame({const.regioncontext_id_field_name: entity_ids, const.regioncontext_context_field_name: entity_context, const.regioncontext_geometry_field_name: entity_coords, const.spabert_emb_field_name: entity_emb})
            self.gdf = gpd.GeoDataFrame(df, geometry=const.regioncontext_geometry_field_name, crs="EPSG:4326")
            self.gdf.to_csv(self.csv_file_path, index=False)
            return self.gdf
        except Exception as e:
            logging.error(f"Error predicting data: {e}")
        return None

def main():
    spabert_trainer = SpaBERTTrainer()
    spabert_trainer.train_model(json_file_path='/home/yaoyi/projects/RegionContext/data/irbid/novateur-poi-sample_spabert_v2_nn100_sdm100.json',
                                model_save_dir='/home/yaoyi/projects/RegionContext/model_weights/',
                                epochs=1)
    spabert_trainer.predict(json_file_path='/home/yaoyi/projects/RegionContext/data/irbid/novateur-poi-sample_spabert_v2_nn100_sdm100.json',
                            model_save_dir='/home/yaoyi/projects/RegionContext/model_weights/',
                            csv_file_path='/home/yaoyi/projects/RegionContext/data/irbid/novateur-poi-sample_spabert_v2_nn100_sdm100_embedding.csv')
    pass
if __name__ == "__main__":
    main()
