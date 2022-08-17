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
filepath = "/model"

st.title("Online Ads Social Issues French Classifier")


# functions that will be called 
def load_model(filepath):
  
  model = AutoModelForSequenceClassification.from_pretrained(filepath)
  return model


def clean_text(sentences):
  st.write('Text cleaning ...')
  cleaned_sentences = []
  for sentence in sentences:
    sentence = re.sub(r"(?:\@|https?\://)\S+", "", sentence)  #remove links and mentions
    sentence = re.sub('\n',r"",sentence)
    cleaned_sentences.append(emoji.replace_emoji(sentence, replace=r"")) #remove emoji
  st.write('Cleaning DONE.')
  return cleaned_sentences

# some ads can still be written in another language than french so 
# it is important to translate to french
def translate_text(sentences):
  st.write('Translation of any text that is not in french language ...')
  french_sentences = []
  translator = Translator()
  for sentence in sentences:
      if(translator.detect(sentence).lang != "fr"):
        french_sentences.append(translator.translate(sentence, dest='fr').text)
      else:
        french_sentences.append(sentence)
  st.write('Translation DONE.')
  return french_sentences


#tokenization of ads before predictions
def preprocess_data(inputs):
  
  # take a batch of texts
  text = inputs["ad_creative_body_new"]
  # encode them
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

st.markdown(' **Interested in classifying online ads according to the social issues discussed ?**')
st.markdown('Upload your file :sunglasses: ')
uploaded_file = st.file_uploader("")


if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  df.drop('Unnamed: 0',axis=1,inplace=True, )
  st.write("length:"+str(len(df)))
  df = df.dropna(subset="ad_creative_body")
  st.write("length no na:"+str(len(df)))
  st.write(df)
  

  #translating ads
  # df['ad_creative_body_new'] = translate_text(df['ad_creative_body'])
  #cleaning ads
  
  df['ad_creative_body_new'] = clean_text(df['ad_creative_body'])

  dt = Dataset.from_pandas(df)

  labels =  [ 'Affaires internationales', 'Energie',
       'Opérations gouvernementales', 'Politique culturelle',
       'Politique sociale', 'Santé',
       'Droits de l’homme libertés publiques et discriminations',
       'Environnement', 'Economic']

  #tokenization 
  st.write('Text tokenization ...')
  encoded_dataset = dt.map(preprocess_data, batched=True, remove_columns=dt.column_names)
  st.write('Tokenization DONE.')
  #loading the fine-tuned model
  model = load_model(filepath)

  test_trainer = Trainer(model) 
  #load the trainer state
  test_trainer_state = pd.read_json(filepath+"/trainer_state.json")
  raw_pred, _, _ = test_trainer.predict(encoded_dataset)  
 
  thresholds = test_trainer_state['log_history'][7]['eval_thresholds']
  n_classes = len(thresholds)
  for i in range(n_classes):
    if(thresholds[i]>0.95) or (thresholds[i]<0.5):
        thresholds[i] = 0.5
  st.write(thresholds)
  #store the predictions in a dataframe 
  y_pred = []
  y_pred_labels = []
  for output in raw_pred:
      sigmoid = torch.nn.Sigmoid()
      probs = sigmoid(torch.from_numpy(output))
      predictions = np.zeros(probs.shape)
      #predictions[np.where(probs >= 0.5)] = 1
      for i in range(n_classes):
          if(probs[i] >= thresholds[i]):
              predictions[i] = 1
      i = 0
      y_pred_label = []
      for pred in predictions:
        if pred == 1 :
          y_pred_label.append(labels[i])
        i += 1
      y_pred_labels.append(y_pred_label)
      y_pred.append(predictions)
  df['social_issues_9cat'] =  y_pred_labels
  df.drop('ad_creative_body_new',axis=1,inplace=True, )
  st.markdown('Check your file with the predicted social issues :smile: ')

  st.write(df) 
  
  csv = convert_df(df)

  st.download_button(
      label="Download the new file as CSV",
      data=csv,
      file_name='ads_with_social_issues.csv',
      mime='text/csv',
  )



