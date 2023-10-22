import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import torch
import warnings
import pydot
from IPython.display import Image, display

class DataFrame(pd.DataFrame):
    _model = None
    _tokenizer = None
    _embmodel = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, texts=None):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          self.texts = texts
          self["text"] = texts
          self.perspectives_loaded = False
          self.speaker_embs = []
          self.emotion_embs = []
          self.object_embs = []
          self.reason_embs = []

    def __getitem__(self, item):
        data = super().__getitem__(item)
        with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          for key in self.columns:
              if isinstance(getattr(self, key + "_embs", None), np.ndarray):
                emb = [getattr(self, key + "_embs", None)[i] for i in data.index.values.tolist()]
              else:
                emb = []
              setattr(data, key + "_embs", emb)

        if isinstance(data, pd.DataFrame):
            data.__class__ = DataFrame
            data._model = self._model
            data._tokenizer = self._tokenizer
            data._embmodel = self._embmodel
            data.device = self.device
            data.perspectives_loaded = self.perspectives_loaded
        return data

    def model_perspectives(self, texts, batch_size=8):
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        all_output = []
        torch.cuda.empty_cache()

        for batch in tqdm(batches):
            batch_prompts = [f"Describe the perspective of this text: {text}" for text in batch]

            input_ids = self._tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True).input_ids.to(self.device)
            output = self._model.generate(input_ids)

            batch_reply = self._tokenizer.batch_decode(output, skip_special_tokens=True)

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
        while self._model is None:
            model_id = "helliun/bart-perspectives"
            self.__class__._model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device)
            self.__class__._tokenizer = AutoTokenizer.from_pretrained(model_id)
            self._model = self.__class__._model
            self._tokenizer = self.__class__._tokenizer

        if not self.perspectives_loaded:
          perspectives = self.model_perspectives(self["text"].tolist(), batch_size=batch_size)
          self["perspectives"] = perspectives
          self["speaker"] = self["perspectives"].apply(lambda x: x["speaker"])
          self["emotion"] = self["perspectives"].apply(lambda x: x["emotion"])
          self["object"] = self["perspectives"].apply(lambda x: x["object"])
          self["reason"] = self["perspectives"].apply(lambda x: x["reason"])
          self.drop(columns=["perspectives"], inplace=True)
          self.perspectives_loaded = True

    def search(self, *args, **kwargs):
        if self._embmodel is None:
            model_id = "sentence-transformers/all-mpnet-base-v1"
            self.__class__._embmodel = SentenceTransformer(model_id).to(self.device)
            self._embmodel = self.__class__._embmodel

        search_df = self.copy()
        search_df["sim_score"] = [0.0]*len(search_df)
        search_df["text"] = self["text"].tolist()

        for key, value in kwargs.items():
            if key == "obj":
              key = "object"
            if value is not None:
                if not isinstance(getattr(self, key + "_embs", None),np.ndarray):
                  if getattr(self, key + "_embs", None) in [None,[]]:
                      setattr(self, key + "_embs", self._embmodel.encode(self[key].tolist()))

                query_emb = self._embmodel.encode(value)
                cos_scores = list(util.cos_sim(query_emb, getattr(self, key + "_embs"))[0])
                search_df["sim_score"] = [sim+score for sim, score in zip(search_df["sim_score"], cos_scores)]

        if any(search_df["sim_score"]):
            search_df.sort_values(by="sim_score", ascending=False, inplace=True)
        
        search_df.drop(columns=["sim_score"], inplace=True)

        return_df =  DataFrame(texts=search_df["text"].tolist(),
                               data={col: search_df[col].tolist() for col in search_df.columns})

        return return_df
    
    def relevant(self):
        if self._embmodel is None:
            model_id = "sentence-transformers/all-mpnet-base-v1"
            self.__class__._embmodel = SentenceTransformer(model_id).to(self.device)
            self._embmodel = self.__class__._embmodel

        search_df = self.copy()
        search_df["sim_score"] = [0.0]*len(search_df)
        search_df["text"] = self["text"].tolist()

        for key in [ "speaker", "emotion", "object", "reason"]:
            if not isinstance(getattr(self, key + "_embs", None),np.ndarray):
              if getattr(self, key + "_embs", None) in [None,[]]:
                  setattr(self, key + "_embs", self._embmodel.encode(self[key].tolist()))

            query_emb = np.average(getattr(self, key + "_embs", None), axis=0)
            cos_scores = list(util.cos_sim(query_emb, getattr(self, key + "_embs"))[0])
            search_df["sim_score"] = [sim+score for sim, score in zip(search_df["sim_score"], cos_scores)]

        if any(search_df["sim_score"]):
            search_df.sort_values(by="sim_score", ascending=False, inplace=True)
        
        search_df.drop(columns=["sim_score"], inplace=True)

        return_df =  DataFrame(texts=search_df["text"].tolist(),
                               data={col: search_df[col].tolist() for col in search_df.columns})

        return return_df

    def view_pydot(self,pdot):
        plt = Image(pdot.create_png())
        display(plt)

    def graph(self, speaker="Speaker", emotion_limit=5, obj_limit=3):
        if self._embmodel is None:
            self.load_embmodel()

        dictionary = {}

        if len(self[self["speaker"] == speaker]) != 0:
          for row in self[self["speaker"] == speaker].iterrows():
            if row[1]["emotion"] in dictionary:
              dictionary[row[1]["emotion"]].append(row[1]["object"])
            else:
              dictionary[row[1]["emotion"]] = [row[1]["object"]]
        else:
          print("Speaker not found")
          return

        graph = pydot.Dot(graph_type='graph')

        speaker_node = pydot.Node(speaker, style="filled", fillcolor="blue")
        graph.add_node(speaker_node)

        happy_embedding = self._embmodel.encode([f'{speaker} is happy'])[0]
        angry_embedding = self._embmodel.encode([f'{speaker} is angry'])[0]

        for emotion in list(dictionary.keys())[:emotion_limit]:
            if emotion != 'speaker':
                emotion_embedding = self._embmodel.encode([f'{speaker} is {emotion}'])[0]
                happy_sim = util.cos_sim([happy_embedding], [emotion_embedding])[0][0]
                angry_sim = util.cos_sim([angry_embedding], [emotion_embedding])[0][0]
                similarity = np.clip(happy_sim - 0.5*angry_sim, 0,1)

                color_value = int(255 * (1 - similarity))

                color_hex = format(color_value, '02x')

                color = f"#{color_hex}ff00" if similarity >= 0.4 else f"#ff{color_hex}00"

                emotion_node = pydot.Node(emotion, style="filled", fillcolor=color)
                graph.add_node(emotion_node)
                graph.add_edge(pydot.Edge(speaker_node, emotion_node))

                for object_ in dictionary[emotion][:obj_limit]:
                    object_node = pydot.Node(object_, style="filled", fillcolor="yellow")
                    graph.add_node(object_node)
                    graph.add_edge(pydot.Edge(emotion_node, object_node))

        self.view_pydot(graph)