import flask
from flask import  render_template, redirect, url_for
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import	TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = flask.Flask(__name__ )

df = pd.read_csv('./model/df_final.csv')

count = CountVectorizer(stop_words='english')
tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(df['Resume'])
count_matrix = count.fit_transform(df['Resume'])

cosine_sim1 = cosine_similarity(tfidf_matrix, tfidf_matrix)
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

df = df.reset_index()
indices = pd.Series(df.index, index=df['Topic'])
all_topics = [df['Topic'][i] for i in range(len(df['Topic']))]

def get_recommendations_byDoc(docid):

    # Get the pairwsie similarity scores of all Resumes with that movie
    sim_scores = list(enumerate(cosine_sim1[docid]))
   
    # Sort the Resumes based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the 10 most similar Resumes
    sim_scores = sim_scores[1:11]
    
    # Get the movie indices
    resume_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar Resumes
    return df[['Document_No','Topic', 'Keywords', 'Resume']].iloc[resume_indices]

def get_recommendations_byTopic(topic):
    recommend = df[ df.Topic == topic]
    return recommend[['Topic', 'Keywords']].iloc[:11]


# Set up the main route
@app.route('/', methods=['GET', 'POST'])

def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))
            
    if flask.request.method == 'POST':
        m_name = flask.request.form['Topics']
        m_name = m_name.title()
#        check = difflib.get_close_matches(m_name,all_titles,cutout=0.50,n=1)
        if m_name not in all_topics:
            return(flask.render_template('Notfound.html',name=m_name))
        else:
            result_final = get_recommendations_byTopic(m_name)
            topics = []
            resume = []
            for i in range(len(result_final)):
                topics.append(result_final.iloc[i][0])
                resume.append(result_final.iloc[i][1])

            return flask.render_template('Result.html',Topics=topics, Resumes=resume,search_name=m_name)



         

if __name__ == '__main__':
    app.run()