import pandas as pd
import numpy as np
import streamlit as st
import torch
import emoji , re

# ML stuff
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from datasets import Dataset, DatasetDict
from googletrans import Translator

modelname = 'flaubert/flaubert_base_cased' 
tokenizer = AutoTokenizer.from_pretrained(modelname)
filepath = "./model"

st.title("Online Ads Social Issues French Classifier")


# functions that will be called 
def load_model(filepath):
  
  model = AutoModelForSequenceClassification.from_pretrained(filepath)
  return model


def clean_text(sentence):
    sentence = re.sub(r"(?:\@|https?\://)\S+", "", sentence)  #remove links and mentions
    sentence = re.sub('\n',r"",sentence)
    return emoji.replace_emoji(sentence, replace=r"") #remove emoji

# some ads can still be written in another language than french so 
# it is important to translate to french
def translate_text(sentence):
    translator = Translator()
    if(translator.detect(sentence).lang != "fr"):
        french_sentence = translator.translate(sentence, dest='fr').text
    else:
        french_sentence = sentence
    return french_sentence


#tokenization of ads before predictions
def preprocess_data(text):
  text = text['0']
  encoding = tokenizer(text, 
                       add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                       padding="max_length", 
                       truncation=True, 
                       max_length=512)
  return encoding

@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv()

st.markdown(' **Interested in knowing what social issues an online ad is discussing ?**')
st.markdown('Write the ad message :sunglasses: then press Ctrl + ')
adText = '''[Anglais] Retrouvez notre dossier pédagogique  consacré au  film UN FILS DU SUD qui revient sur les "freedom rides"  et le mouvement des droits civiques  : Repères historiques Entretien avec l'historien Nicolas Martin-Breteau   Activités pédagogiques Anglais (Collège, Lycée) Corrigé des activités'''
adText = st.text_area('', adText)

st.markdown('Classification in process ...')

#translating ads
adTextNew = translate_text(adText)

#cleaning ads
adTextNew = clean_text(adTextNew)

labels =  [ 'Affaires internationales', 'Energie',
        'Opérations gouvernementales', 'Politique culturelle',
        'Politique sociale', 'Santé',
        'Droits de l’homme libertés publiques et discriminations',
        'Environnement', 'Economic']
df = []
for i in range(2):
        df.append(adTextNew)
df = pd.DataFrame(df)
dt = Dataset.from_pandas(df)
    #tokenization 
encoded_text = dt.map(preprocess_data, batched=True, remove_columns=dt.column_names)

    #loading the fine-tuned model
model = load_model(filepath)

test_trainer = Trainer(model) 
    #load the trainer state
test_trainer_state = pd.read_json(filepath+"/trainer_state.json")
raw_pred, _, _ = test_trainer.predict(encoded_text)  
    
thresholds = test_trainer_state['log_history'][7]['eval_thresholds']

    #store the predictions in a dataframe 
output = raw_pred[0]
sigmoid = torch.nn.Sigmoid()
probs = sigmoid(torch.from_numpy(output))
predictions = np.zeros(probs.shape)
predictions[np.where(probs >= 0.5)] = 1
i = 0
y_pred_label = []
for pred in predictions:
        if pred == 1 :
            y_pred_label.append({"social_issue":labels[i], "probability":probs[i].item()})
        i += 1

st.markdown('Here are the classification results:')
    # st.write(probs)
st.write(y_pred_label)
  



