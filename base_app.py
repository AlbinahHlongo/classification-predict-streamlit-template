
#streamlit dependencies

import streamlit as st
import joblib, os

## data dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from textblob import TextBlob
import re
import sklearn
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
#from nlppreprocess import NLP # pip install nlppreprocess
#import en_core_web_sm
from nltk import pos_tag

import base64

import seaborn as sns
import re

from wordcloud import WordCloud

from nlppreprocess import NLP
nlp = NLP()
st.set_option('deprecation.showPyplotGlobalUse', False)


def cleaner(line):

    # Removes RT, url and trailing white spaces
    line = re.sub(r'^RT ','', re.sub(r'https://t.co/\w+', '', line).strip()) 

    # Removes puctuation
    punctuation = re.compile("[.;:!\'’‘“”?,\"()\[\]]")
    tweet = punctuation.sub("", line.lower()) 

    # Removes stopwords
    nlp_for_stopwords = NLP(replace_words=True, remove_stopwords=True, remove_numbers=True, remove_punctuations=False) 
    tweet = nlp_for_stopwords.process(tweet) # This will remove stops words that are not necessary. The idea is to keep words like [is, not, was]
    # https://towardsdatascience.com/why-you-should-avoid-removing-stopwords-aa7a353d2a52

    # tokenisation
    # We used the split method instead of the word_tokenise library because our tweet is already clean at this point
    # and the twitter data is not complicated
    tweet = tweet.split() 

    # POS 
    pos = pos_tag(tweet)


    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tweet = ' '.join([lemmatizer.lemmatize(word, po[0].lower()) if po[0].lower() in ['n', 'r', 'v', 'a'] else word for word, po in pos])
    # tweet = ' '.join([lemmatizer.lemmatize(word, 'v') for word in tweet])

    return tweet


##reading in the raw data and its cleaner

vectorizer = open('resources/tfidfvect.pkl','rb')   ##  will be replaced by the cleaning and preprocessing function
tweet_cv = joblib.load(vectorizer)

#@st.cache
data = pd.read_csv('resources/train1.csv')

def main():
    """Tweets classifier App"""
 
    st.title('Decoding Digital Dialogue  :)')

    from PIL import Image
    image = Image.open('resources/imgs/horizon.png')

    st.image(image, caption='#DecodingDigitalDialogue', use_column_width=True)

    st.subheader('Climate Change Belief Analysis: Based on Tweets')
    

    ##creating a sidebar for selection purposes


    pages = ['Info', 'Visuals', 'Predictions', 'Contact Us']

    selection = st.sidebar.radio('Menu', pages)

    #st.sidebar.image(image, caption='Which Tweet are you?', use_column_width=True)



    ##information page

    if selection == 'Info':
        st.info('General Information')
        st.write('Decoding Digital Dialogue!!!')
        st.markdown(""" Our task is to interpret model insights; 
                    beyond accuracy, we'll dig deep into the features 
                    that play a pivotal role in our model's decision-making process. 
                    This understanding is crucial for refining our approach and 
                    enhancing the effectiveness of our application. 
        """)


        raw = st.checkbox('See raw data')
        if raw:
            st.dataframe(data.head(25))

    ## Charts page

    if selection == 'Visuals':
        st.info('The sentiment class distribution in a dataset indicates the frequency or count of different sentiment categories assigned to instances of text data. Analyzing the sentiment class distribution will helps us identify potential challenges, such as imbalanced classes.')


       # Number of Messages Per Sentiment
      
        # Labeling the target
        data['sentiment'] = [['Anti', 'Neutral', 'Pro', 'Factual'][x+1] for x in data['sentiment']]
        
        # checking the distribution
     
        st.markdown("<h1 style='text-align: center; font-size: 1.5em;'>Sentiment Class Distribution</h1>", unsafe_allow_html=True)
        values = data['sentiment'].value_counts()/data.shape[0]
        labels = (data['sentiment'].value_counts()/data.shape[0]).index
        colors = ['lightgreen', 'lightskyblue', 'lightgrey', 'pink']
        plt.pie(x=values, labels=labels, autopct='%1.0f%%', startangle=90, explode= (0.1, 0.1, 0.1, 0.1), colors=colors)
        
        st.pyplot()
        st.write('     ')
        st.write('     ')

        
 
        
        # checking the distribution
        st.markdown("<h1 style='text-align: center; font-size: 1.5em;'>Number of Messages Per Sentiment</h1>", unsafe_allow_html=True)
        sns.countplot(x='sentiment' ,data = data, palette='PRGn')
        plt.ylabel('Count')
        plt.xlabel('Sentiment')
        
        st.pyplot()

        # Popular Tags
        st.write('     ')
        st.markdown("<h1 style='text-align: center; font-size: 1.5em;'>Popular tags found in the tweets</h1>", unsafe_allow_html=True)
        data['users'] = [''.join(re.findall(r'@\w{,}', line)) if '@' in line else np.nan for line in data.message]
        sns.countplot(y="users", hue="sentiment", data=data,
                    order=data.users.value_counts().iloc[:20].index, palette='PRGn') 
        plt.ylabel('User')
        plt.xlabel('Number of Tags')
        plt.title('Top 20 Most Popular Tags')
        st.pyplot()


        st.markdown("<h1 style='text-align: center; font-size: 1.5em;'>Trending Neutral Climate Change Hashtags (#)</h1>", unsafe_allow_html=True)
        
        from PIL import Image
        image = Image.open('resources/imgs/work-clouds.png')
        st.image(image,  use_column_width=True)

        st.markdown("<h1 style='text-align: center; font-size: 1.5em;'>Trending Pro climate change hashtags(#)</h1>", unsafe_allow_html=True)
        
        from PIL import Image
        image = Image.open('resources/imgs/pro-hashtags.png')
        st.image(image,  use_column_width=True)

        st.markdown("<h1 style='text-align: center; font-size: 1.5em;'>Trending Factual Climate Change Hashtags (#)</h1>", unsafe_allow_html=True)
     
        from PIL import Image
        image = Image.open('resources/imgs/news-hashtags.png')
        st.image(image,  use_column_width=True)

        st.markdown("<h1 style='text-align: center; font-size: 1.5em;'>Trending Neutral Climate Change Hashtags (#)</h1>", unsafe_allow_html=True)
        from PIL import Image
        image = Image.open('resources/imgs/neutral-hashtags.png')
        st.image(image,  use_column_width=True)

        st.markdown("<h1 style='text-align: center; font-size: 1.5em;'>Trending ANTI Climate Change Hashtags (#)</h1>", unsafe_allow_html=True)
       
        from PIL import Image
        image = Image.open('resources/imgs/anti-hashtags.png')
        st.image(image,  use_column_width=True)


        

    ## prediction page

    if selection == 'Predictions':

        st.info('Let us predict your tweets using our model')

        data_source = ['Select option', 'Single text', 'Dataset'] ## differentiating between a single text and a dataset inpit

        source_selection = st.selectbox('What to classify?', data_source)

        # Load Our Models
        def load_prediction_models(model_file):
            loaded_models = joblib.load(open(os.path.join(model_file),"rb"))
            return loaded_models

        # Getting the predictions
        def get_keys(val,my_dict):
            for key,value in my_dict.items():
                if val == value:
                    return key


        if source_selection == 'Single text':
            ### SINGLE TWEET CLASSIFICATION ###
            st.subheader('Single tweet classification')

            input_text = st.text_area('Enter Text (max. 140 characters):') ##user entering a single text to classify and predict
            all_ml_models = ["LogisticReg","RandomFOREST","Vectorizer","SupportVectorMachine"]
            model_choice = st.selectbox("Choose ML Model",all_ml_models)

            st.info('for more information on the above ML Models please visit: https://datakeen.co/en/8-machine-learning-algorithms-explained-in-human-language/')

            prediction_labels = {'Negative':-1,'Neutral':0,'Positive':1,'News':2}
            if st.button('Classify'):

                st.text("Original test ::\n{}".format(input_text))
                text1 = cleaner(input_text) ###passing the text through the 'cleaner' function
                vect_text = tweet_cv.transform([text1]).toarray()
                if model_choice == 'LogisticReg':
                    predictor = load_prediction_models("resources/Logistic_regression.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
                elif model_choice == 'RandomFOREST':
                    predictor = load_prediction_models("resources/Random_model.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
                elif model_choice == 'Vectorizer':
                    predictor = load_prediction_models("resources/tfidfvect.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
                elif model_choice == 'SupportVectorMachine':
                    predictor = load_prediction_models("resources/svm_model1.pkl")
                    prediction = predictor.predict(vect_text)
				# st.write(prediction)

                final_result = get_keys(prediction,prediction_labels)
                st.success("Tweet Categorized as:: {}".format(final_result))

        if source_selection == 'Dataset':
            ### DATASET CLASSIFICATION ###
            st.subheader('Dataset tweet classification')

            all_ml_models = ["LogisticReg","RandomFOREST","Vectorizer","SupportVectorMachine"]
            model_choice = st.selectbox("Choose ML Model",all_ml_models)

            st.info('for more information on the above ML Models please visit: https://datakeen.co/en/8-machine-learning-algorithms-explained-in-human-language/')


            prediction_labels = {'Negative':-1,'Neutral':0,'Positive':1,'News':2}
            text_input = st.file_uploader("Choose a CSV file", type="csv")
            if text_input is not None:
                text_input = pd.read_csv(text_input)

            #X = text_input.drop(columns='tweetid', axis = 1, inplace = True)   

            uploaded_dataset = st.checkbox('See uploaded dataset')
            if uploaded_dataset:
                st.dataframe(text_input.head(25))

            col = st.text_area('Enter column to classify')

            #col_list = list(text_input[col])

            #low_col[item.lower() for item in tweet]
            #X = text_input[col]

            #col_class = text_input[col]
            
            if st.button('Classify'):

                st.text("Original test ::\n{}".format(text_input))
                X1 = text_input[col].apply(cleaner) ###passing the text through the 'cleaner' function
                X1_lower = X1.str.lower()
                vect_text = tweet_cv.fit.transform([X1_lower]).toarray()
                if model_choice == 'LogisticReg':
                    predictor = load_prediction_models("resources/Logistic_regression.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
                elif model_choice == 'RandomFOREST':
                    predictor = load_prediction_models("Random_model.pkl")
                    prediction = predictor.predict(vect_text)
                    # st.write(prediction)
               
                elif model_choice == 'SupportVectorMachine':
                    predictor = load_prediction_models("resources/svm_model1.pkl")
                    prediction = predictor.predict(vect_text)

                elif model_choice == 'Vectorizer':
                    predictor = load_prediction_models("resources/tfidfvect.pkl")
                    prediction = predictor.predict(vect_text)


                
				# st.write(prediction)
                text_input['sentiment'] = prediction
                final_result = get_keys(prediction,prediction_labels)
                st.success("Tweets Categorized as:: {}".format(final_result))

                
                csv = text_input.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
                href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'

                text_input['sentiment']=text_input['sentiment'].apply(lambda x: x.lower())

                st.markdown(href, unsafe_allow_html=True)


    ##contact page
    if selection == 'Contact Us':

        st.markdown(
    """
    Horizon News, DDD isn’t just a tool, it is a game changer because it sifts through the noise chatter and hands you insights on a silver platter. The channel will be able to steer the news coverage in real-time by reflecting public sentiment. Imagine being steps ahead in reporting climate issues, shaping discussions, and keeping the audience engaged.
    """
)

        st.info("For any queries, don't hesitate to contact us.")

        contact_details = [
    "Sonkhe: [shongwesonkhe@gmail.com](mailto:shongwesonkhe@gmail.com)",
    "Albinah: [albinahlongo@gmail.com](mailto:albinahlongo@gmail.com)",
    "Ntebatse: [rachidintebatse@gmail.com](mailto:rachidintebatse@gmail.com)",
    "Millicent: [millicenttsweleng@gmail.com](mailto:millicenttsweleng@gmail.com)",
    "Eugen: [sibandaeugene12345@gmail.com](mailto:sibandaeugene12345@gmail.com)",
    "Terrence: [terrencerivisi@gmail.com](mailto:terrencerivisi@gmail.com)",
]

        for contact in contact_details:
            st.write(contact)

        

        # Footer 
        # Footer 
        image = Image.open('resources/imgs/footer.jpeg')

        st.image(image, caption='Team  JL2      #DDD  ', use_column_width=True)


if __name__ == '__main__':
	main()