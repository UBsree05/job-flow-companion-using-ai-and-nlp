import os
import pandas as pd
import resparser, match
import nltk
from nltk.corpus import stopwords
import indeed_web_scraping_using_bs4

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('maxent_ne_chunker')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('brown')

stopw  = set(stopwords.words('english'))

def find_sort_job(f):
    job = pd.read_csv(r'C:\Users\Kowshik\PycharmProjects\resume-job-recommendation-systemss\resume_screening\indeed_data.csv')
    job['test'] = job['description'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stopw)]))
    df = job.drop_duplicates(subset='test').reset_index(drop=True)
    df['clean'] = df['test'].apply(match.preprocessing)
    jobdesc = (df['clean'].values.astype('U'))
    skills = resparser.skill(f)
    # skills = ' '.join(word for word in skills['skills'])
    skills = match.preprocessing(skills[0])
    # del skills[0]
    count_matrix = match.vectorizing(skills, jobdesc)
    matchPercentage = match.coSim(count_matrix)
    matchPercentage = pd.DataFrame(matchPercentage, columns=['Skills Match'])
    #Job Vacancy Recommendations
    result_cosine = df[['title','company','link']]
    result_cosine = result_cosine.join(matchPercentage)
    result_cosine = result_cosine[['title','company','Skills Match','link']]
    result_cosine.columns = ['Job Title','Company','Skills Match','Link']
    result_cosine = result_cosine.sort_values('Skills Match', ascending=False).reset_index(drop=True).head(20)
    return result_cosine

    
abc = find_sort_job(r'instance\resume_files\KOWSHIK.pdf')
print(abc)