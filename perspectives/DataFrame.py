import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import torch
import warnings

class DataFrame(pd.DataFrame):
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, texts=None):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          self.texts = texts
          self["texts"] = self.texts
          self.tokenizer = None
          self.model = None
          self.embmodel = None
          self.perspectives_loaded = False
          self.device = "cuda" if torch.cuda.is_available() else "cpu"
          self.speaker_embs = []
          self.emotion_embs = []
          self.object_embs = []
          self.reason_embs = []
    
    def load_model(self):
        model_id = "helliun/bart-perspectives"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device)
    
    def load_embmodel(self):
        model_id = "sentence-transformers/all-mpnet-base-v1"
        self.embmodel = SentenceTransformer(model_id).to(self.device)
    
    def silly(self):
       return self.head()
    
    def get_perspective(self,text):
        prompt = f"Describe the perspective of this text: {text}"
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)
        output = self.model.generate(input_ids)
        reply = self.tokenizer.batch_decode(output)[0]
        reply = reply.replace("<pad>","").replace("</s>","")
        reply = reply[reply.find("Speaker:"):]
        out_list = reply.split("\n")
        speaker = out_list[0][9:]
        emotion = out_list[1][3:]
        obj = out_list[2][6:]
        reason = out_list[3][8:]
        reason = reason[:reason.find("Speaker:")-1]
        return {"speaker": speaker, "emotion": emotion, "object": obj, "reason": reason}
        return reply
    
    def model_perspectives(self, texts, batch_size=8):
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        all_output = []
        torch.cuda.empty_cache()

        for batch in tqdm(batches):
            batch_prompts = [f"Describe the perspective of this text: {text}" for text in batch]

            input_ids = self.tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True).input_ids.to(self.device)
            output = self.model.generate(input_ids)

            batch_reply = self.tokenizer.batch_decode(output, skip_special_tokens=True)

            for reply in batch_reply:
                reply = reply[reply.find("Speaker:"):]
                try:
                  out_list = reply.split("\n")
                  speaker = out_list[0][9:]
                  emotion = out_list[1][3:]
                  obj = out_list[2][6:]
                  reason = out_list[3][8:]
                  reason = reason[:reason.find("Speaker:")-1] if "Speaker:" in reason else reason
                  all_output.append({"speaker": speaker, "emotion": emotion, "object": obj, "reason": reason})
                except:
                  all_output.append({"speaker": "", "emotion": "", "object": "", "reason": ""})

        return all_output

    def get_perspectives(self, batch_size=8):
        if self.model == None:
          self.load_model()
        if not self.perspectives_loaded:
          perspectives = self.model_perspectives(self.texts, batch_size=batch_size)
          self["perspectives"] = perspectives
          self["speaker"] = self["perspectives"].apply(lambda x: x["speaker"])
          self["emotion"] = self["perspectives"].apply(lambda x: x["emotion"])
          self["object"] = self["perspectives"].apply(lambda x: x["object"])
          self["reason"] = self["perspectives"].apply(lambda x: x["reason"])
          self.drop(columns=["perspectives"], inplace=True)
          self.perspectives_loaded = True

    def search(self, speaker=None, emotion=None, obj=None, reason=None):
        if self.embmodel == None:
          self.load_embmodel()

        search_df = self.copy()
        search_df["sim_score"] = [0.0]*len(search_df)
        search_df["text"] = self.texts

        if speaker != None:
          if self.speaker_embs == []:
            speaker_embs = self.embmodel.encode(self["speaker"])
            self.speaker_embs = speaker_embs
          query = speaker
          query_emb = self.embmodel.encode(query)
          cos_scores = list(util.cos_sim(query_emb, self.speaker_embs)[0])
          search_df["sim_score"] = [sim+score for sim, score in zip(search_df["sim_score"],cos_scores)]

        if emotion != None:
          if self.emotion_embs == []:
            emotion_embs = self.embmodel.encode(self["emotion"])
            self.emotion_embs = emotion_embs
          query = emotion
          query_emb = self.embmodel.encode(query)
          cos_scores = list(util.cos_sim(query_emb, self.emotion_embs)[0])
          search_df["sim_score"] = [sim+score for sim, score in zip(search_df["sim_score"],cos_scores)]

        if obj != None:
          if self.object_embs == []:
            object_embs = self.embmodel.encode(self["object"])
            self.object_embs = object_embs
          query = obj
          query_emb = self.embmodel.encode(query)
          cos_scores = list(util.cos_sim(query_emb, self.object_embs)[0])
          search_df["sim_score"] = [sim+score for sim, score in zip(search_df["sim_score"],cos_scores)]
        
        if reason != None:
          if self.reason_embs == []:
            reason_embs = self.embmodel.encode(self["reason"])
            self.reason_embs = reason_embs
          query = reason
          query_emb = self.embmodel.encode(query)
          cos_scores = list(util.cos_sim(query_emb, self.reason_embs)[0])
          search_df["sim_score"] = [sim+score for sim, score in zip(search_df["sim_score"],cos_scores)]
        
        if search_df["sim_score"].tolist() != [0.0]*len(self):
          search_df.sort_values(by="sim_score", ascending=False, inplace=True)

        search_df.drop(columns=["sim_score"], inplace=True)

        return DataFrame(texts=search_df["text"].tolist(),data={"texts":search_df["text"].tolist(),"speaker":search_df["speaker"].tolist(),
                                                         "emotion":search_df["emotion"].tolist(),
                                                         "object":search_df["object"].tolist(),
                                                         "reason":search_df["reason"].tolist()
                                                         },)

