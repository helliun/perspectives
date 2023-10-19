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
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, texts=None, embmodel=None, model=None):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          self.texts = texts
          self["text"] = self.texts
          self.tokenizer = None
          self.model = model
          self.embmodel = embmodel
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

    def search(self, *args, **kwargs):
        if self.embmodel is None:
            self.load_embmodel()

        search_df = self.copy()
        search_df["sim_score"] = [0.0]*len(search_df)
        search_df["text"] = self.texts

        for key, value in kwargs.items():
            if key == "obj":
              key = "object"
            if value is not None:
                if not isinstance(getattr(self, key + "_embs", None),np.ndarray):
                  if getattr(self, key + "_embs", None) in [None,[]]:
                      setattr(self, key + "_embs", self.embmodel.encode(self[key]))

                query_emb = self.embmodel.encode(value)
                cos_scores = list(util.cos_sim(query_emb, getattr(self, key + "_embs"))[0])
                search_df["sim_score"] = [sim+score for sim, score in zip(search_df["sim_score"], cos_scores)]

        if any(search_df["sim_score"]):
            search_df.sort_values(by="sim_score", ascending=False, inplace=True)
        
        search_df.drop(columns=["sim_score"], inplace=True)

        return_df =  DataFrame(texts=search_df["text"].tolist(),
                               data={col: search_df[col].tolist() for col in search_df.columns},
                               embmodel=self.embmodel,
                               model=self.model)

        return return_df

    def view_pydot(self,pdot):
      plt = Image(pdot.create_png())
      display(plt)

    def profile_graph(self, speaker, emotion_limit=-1, obj_limit=-1):
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

        # Create an empty graph
        graph = pydot.Dot(graph_type='graph')

        # Start with the Speaker node
        speaker_node = pydot.Node(speaker, style="filled", fillcolor="blue")
        graph.add_node(speaker_node)

        # Get the embedding for the word "happy"
        happy_embedding = self.embmodel.encode([f'{speaker} is happy'])[0]
        angry_embedding = self.embmodel.encode([f'{speaker} is angry'])[0]


        # Add Emotion nodes
        for emotion in list(dictionary.keys())[:emotion_limit]:
            if emotion != 'speaker':
                # Calculate the cosine similarity to the word "happy"
                emotion_embedding = self.embmodel.encode([f'{speaker} is {emotion}'])[0]
                happy_sim = util.cos_sim([happy_embedding], [emotion_embedding])[0][0]
                angry_sim = util.cos_sim([angry_embedding], [emotion_embedding])[0][0]
                similarity = np.clip(happy_sim - 0.5*angry_sim, 0,1)

                # Convert similarity to an integer between 0 and 255
                color_value = int(255 * (1 - similarity))

                # Convert the color value to hexadecimal
                color_hex = format(color_value, '02x')

                # Define color based on similarity (green for high, red for low)
                color = f"#{color_hex}ff00" if similarity >= 0.4 else f"#ff{color_hex}00"

                emotion_node = pydot.Node(emotion, style="filled", fillcolor=color)
                graph.add_node(emotion_node)
                graph.add_edge(pydot.Edge(speaker_node, emotion_node))

                # Add Object nodes
                for object_ in dictionary[emotion][:obj_limit]:
                    object_node = pydot.Node(object_, style="filled", fillcolor="yellow")
                    graph.add_node(object_node)
                    graph.add_edge(pydot.Edge(emotion_node, object_node))

        # Save the graph to a file
        graph.write_png('graph.png')
        self.view_pydot(graph)