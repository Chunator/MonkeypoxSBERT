###OVERALL Code
import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import urllib.request
from sentence_transformers import SentenceTransformer
import os
from transformers import AutoTokenizer, AutoModel
import nltk
from tqdm import tqdm
from tqdm import trange
import csv

dir1 = '/Users/steven/Desktop/대학원/논문/작업중인 논문/Pox research/' #Address1
dir2 = '/Users/steven/Desktop/' #Address2
File_PATH1 = os.path.join(dir1,'NODUPLICATE-Result.csv') #File Path
train_data = pd.read_csv(File_PATH1) #place "r" before the path string to address special character, such as '\'. 
train_data.head()


###Main Code 1###
stop_words = nltk.corpus.stopwords.words('english')
more_stopwords = []
stop_words = stop_words + more_stopwords

def clean_text(text):
    import re

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''    
    text = str(text).lower()
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('\n', '', text) 
    deltext = ['Ä','ì','Œ','∫','Æ','¬','¸','˛','Ç','◊','µ','ı','˜','Â','%',';','¯','˘','¿','Æ','Ú','Ò','','Ô','Ó','˝','Ï','Î','Í','Å','’','”','∏','Ø','ˆ','¨','Á','ˇ','‰','√','Ø','´','„','Œ','》','“','”','【','】','·','@','Œ','º','œ','å','Ω','≈','ç','√','∫','˜','µ','≤','≥','÷','å','ß','∂','ƒ','˙','˚','¬','…','æ','‘','“','π','ø','ˆ','¨','¥','†','®','´','∑','©']
    text = text.replace('-', ' ')
    for i in deltext:
        text = text.replace(i, '')
    text = re.sub(r'[0-9]+', '', text) 
    return text

def remove_stopwords(text):
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text

lemma = nltk.WordNetLemmatizer()
def lemma_text(text):
    text = ' '.join(lemma.lemmatize(word) for word in text.split(' '))
    return text

def preprocessing(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemma_text(text)
    return text

train_data['Content'] = train_data['Content'].apply(lambda x: preprocessing(x))
#Example)
#A = 'Background: Monkeypox is a poorly described emerging zoonosis endemic to Central and Western Africa. Methods: Using surveillance data from Tshuapa Province, Democratic Republic of the Congo during 2011-2015, we evaluated differences in incidence, exposures, and clinical presentation of polymerase chain reaction-confirmed cases by sex and age. Results: We report 1057 confirmed cases. The average annual incidence was 14.1 per 100 000 (95% confidence interval, 13.3-15.0). The incidence was higher in male patients (incidence rate ratio comparing males to females, 1.21; 95% confidence interval, 1.07-1.37), except among those 20-29 years old (0.70;. 51-.95). Females aged 20-29 years also reported a high frequency of exposures (26.2%) to people with monkeypox-like symptoms.The highest incidence was among 10-19-year-old males, the cohort reporting the highest proportion of animal exposures (37.5%). The incidence was lower among those presumed to have received smallpox vaccination than among those presumed unvaccinated. No differences were observed by age group in lesion count or lesion severity score. Conclusions: Monkeypox incidence was twice that reported during 1980-1985, an increase possibly linked to declining immunity provided by smallpox vaccination. The high proportion of cases attributed to human exposures suggests changing exposure patterns. Cases were distributed across age and sex, suggesting frequent exposures that follow sociocultural norms. © 2021 Published by Oxford University Press for the Infectious Diseases Society of America 2021.'
#print(preprocessing(A))
#print(train_data['Abstract'])
#print("DONE")


###Main Code 2###
#name = ['sentence-transformers/LaBSE', 'sentence-transformers/multi-qa-mpnet-base-dot-v1', 'multi-qa-mpnet-base-dot-v1', 'anferico/bert-for-patents', 'AI-Growth-Lab/PatentSBERTa','all-mpnet-base-v2','sentence-transformers/all-MiniLM-L12-v2', 'all-MiniLM-L6-v2','all-MiniLM-L12-v2','multi-qa-distilbert-cos-v1']
name = ['AI-Growth-Lab/PatentSBERTa']

trial = 1
for names in name:
    print("Trial ",trial, " : ", names)
    model = SentenceTransformer(names) 
    print("Model Embedding Test")
    sentences = ["This is an example sentence", "Each sentence is converted"]
    embeddings = model.encode(sentences)
    print(embeddings)

###Main Code 3###
#Testing
    File_PATH2 = os.path.join(dir1,'SBERT_WOS_Result_'+str(trial)+'.csv') # OS independent way of building paths
    f2 = open(File_PATH2, 'w')
    writer2 = csv.writer(f2)
    writer2.writerow(['Order', 'Title', 'Abstract', 'Content', 'CPC_MAIN', 'CPC_ALL', 'Applicant', 'Holder', 'cos_score','Target'])

    print("DB Embedding Start")
    embresult = []
    for contents in tqdm(train_data['Content']):
        embresult.append(model.encode(contents))
    train_data['embedding'] = embresult
    print("DB Embedding Complete")

    def cos_sim(A, B):
        return dot(A, B)/(norm(A)*norm(B))

    def return_answer(question):
        embedding = model.encode(question)
        train_data['score'] = train_data.apply(lambda x: cos_sim(x['embedding'], embedding), axis=1)
        result = [train_data.loc[train_data['score'].idxmax()]['Order'], train_data.loc[train_data['score'].idxmax()]['Title'], train_data.loc[train_data['score'].idxmax()]['Abstract'], train_data.loc[train_data['score'].idxmax()]['Content'], train_data.loc[train_data['score'].idxmax()]['CPC_MAIN'], train_data.loc[train_data['score'].idxmax()]['CPC_ALL'], train_data.loc[train_data['score'].idxmax()]['Applicant'], train_data.loc[train_data['score'].idxmax()]['Holder'], max(train_data['score'])]
        return result

    File_PATH3 = os.path.join(dir2,'Mainpath_Result.csv') # OS independent way of building paths
    test_data = pd.read_csv(File_PATH3) #place "r" before the path string to address special character, such as '\'. 
    test_data['Content'] = test_data['Content'].apply(lambda x: preprocessing(x))
                     
    print("Target Embedding Start")                
    for target in tqdm(test_data['Content']):
        SBERTresult = return_answer(target)
        SBERTresult.append(target)
        writer2.writerow(SBERTresult)
    trial += 1
    print("Target Embedding Complete")
    f2.close()
print("Done")

