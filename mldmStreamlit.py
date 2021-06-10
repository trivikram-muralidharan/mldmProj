# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 14:24:52 2021

@author: muralidh
"""


import streamlit as st


import warnings
warnings.filterwarnings('ignore')
import sys
import numpy as np
import altair as alt
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

import pickle

model = load_model("lstmFinal.model")

with open("tokenizer.pkl","rb") as f:
    tokenizer = pickle.load(f)

countVectorizer = ""
with open("CountVectorizer.pkl","rb") as f:
    countVectorizer = pickle.load(f)

tfidfVectorizer = ""
with open("TFidfVectorizer.pkl","rb") as f:
    tfidfVectorizer = pickle.load(f)

RF_TFIDF = ""
with open("RF_TFIDF.pkl","rb") as f:
    RF_TFIDF = pickle.load(f)


RF_BOW = ""
with open("RF_BOW.pkl","rb") as f:
    RF_BOW = pickle.load(f)


NB_TFIDF = ""
with open("NaiveTFIDF.pkl","rb") as f:
    NB_TFIDF = pickle.load(f)


NB_BOW = ""
with open("NaiveBOW.pkl","rb") as f:
    NB_BOW = pickle.load(f)


SVM_TFIDF = ""
with open("svmTFIDF.pkl","rb") as f:
    SVM_TFIDF = pickle.load(f)


SVM_BOW = ""
with open("svmBOW.pkl","rb") as f:
    SVM_BOW = pickle.load(f)



def paginator(label, items, items_per_page=10, on_sidebar=True):
    """Lets the user paginate a set of items.
    Parameters
    ----------
    label : str
        The label to display over the pagination widget.
    items : Iterator[Any]
        The items to display in the paginator.
    items_per_page: int
        The number of items to display per page.
    on_sidebar: bool
        Whether to display the paginator widget on the sidebar.
        
    Returns
    -------
    Iterator[Tuple[int, Any]]
        An iterator over *only the items on that page*, including
        the item's index.
    Example
    -------
    This shows how to display a few pages of fruit.
    >>> fruit_list = [
    ...     'Kiwifruit', 'Honeydew', 'Cherry', 'Honeyberry', 'Pear',
    ...     'Apple', 'Nectarine', 'Soursop', 'Pineapple', 'Satsuma',
    ...     'Fig', 'Huckleberry', 'Coconut', 'Plantain', 'Jujube',
    ...     'Guava', 'Clementine', 'Grape', 'Tayberry', 'Salak',
    ...     'Raspberry', 'Loquat', 'Nance', 'Peach', 'Akee'
    ... ]
    ...
    ... for i, fruit in paginator("Select a fruit page", fruit_list):
    ...     st.write('%s. **%s**' % (i, fruit))
    """

    # Figure out where to display the paginator
    if on_sidebar:
        location = st.sidebar.empty()
    else:
        location = st.empty()

    # Display a pagination selectbox in the specified location.
    items = list(items)
    n_pages = len(items)
    n_pages = (len(items) - 1) // items_per_page + 1
    page_format_func = lambda i: "Page %s" % i
    page_number = location.selectbox(label, range(n_pages), format_func=page_format_func)

    # Iterate over the items in the page to let the user display them.
    min_index = page_number * items_per_page
    max_index = min_index + items_per_page
    import itertools
    return itertools.islice(enumerate(items), min_index, max_index)

def main():
    
    
    
    st.write(""" 
    ## Fake News Detector! 
    
    
    """)
    
    
    #sidebar stuff
    options = []
    
    
    st.sidebar.text("Choose from the following options:")
    
    options = ["Demo","Analysis","Metrics"]
    
    selected_page = st.sidebar.selectbox("Select the function", options)
    
    if(selected_page == "Demo"):
        inputText = st.text_input(label="Enter the news article." )
        print(inputText)
        #userInp = "WASHINGTON (Reuters) - The head of a conservative Republican faction in the U.S. Congress, who voted this month for a huge expansion of the national debt to pay for tax cuts, called himself a â€œfiscal conservativeâ€ on Sunday and urged budget restraint in 2018. In keeping with a sharp pivot under way among Republicans, U.S. Representative Mark Meadows, speaking on CBSâ€™ â€œFace the Nation,â€ drew a hard line on federal spending, which lawmakers are bracing to do battle over in January. When they return from the holidays on Wednesday, lawmakers will begin trying to pass a federal budget in a fight likely to be linked to other issues, such as immigration policy, even as the November congressional election campaigns approach in which Republicans will seek to keep control of Congress. President Donald Trump and his Republicans want a big budget increase in military spending, while Democrats also want proportional increases for non-defense â€œdiscretionaryâ€ spending on programs that support education, scientific research, infrastructure, public health and environmental protection. â€œThe (Trump) administration has already been willing to say: â€˜Weâ€™re going to increase non-defense discretionary spending ... by about 7 percent,â€™â€ Meadows, chairman of the small but influential House Freedom Caucus, said on the program. â€œNow, Democrats are saying thatâ€™s not enough, we need to give the government a pay raise of 10 to 11 percent. For a fiscal conservative, I donâ€™t see where the rationale is. ... Eventually you run out of other peopleâ€™s money,â€ he said. Meadows was among Republicans who voted in late December for their partyâ€™s debt-financed tax overhaul, which is expected to balloon the federal budget deficit and add about $1.5 trillion over 10 years to the $20 trillion national debt. â€œItâ€™s interesting to hear Mark talk about fiscal responsibility,â€ Democratic U.S. Representative Joseph Crowley said on CBS. Crowley said the Republican tax bill would require the  United States to borrow $1.5 trillion, to be paid off by future generations, to finance tax cuts for corporations and the rich. â€œThis is one of the least ... fiscally responsible bills weâ€™ve ever seen passed in the history of the House of Representatives. I think weâ€™re going to be paying for this for many, many years to come,â€ Crowley said. Republicans insist the tax package, the biggest U.S. tax overhaul in more than 30 years,  will boost the economy and job growth. House Speaker Paul Ryan, who also supported the tax bill, recently went further than Meadows, making clear in a radio interview that welfare or â€œentitlement reform,â€ as the party often calls it, would be a top Republican priority in 2018. In Republican parlance, â€œentitlementâ€ programs mean food stamps, housing assistance, Medicare and Medicaid health insurance for the elderly, poor and disabled, as well as other programs created by Washington to assist the needy. Democrats seized on Ryanâ€™s early December remarks, saying they showed Republicans would try to pay for their tax overhaul by seeking spending cuts for social programs. But the goals of House Republicans may have to take a back seat to the Senate, where the votes of some Democrats will be needed to approve a budget and prevent a government shutdown. Democrats will use their leverage in the Senate, which Republicans narrowly control, to defend both discretionary non-defense programs and social spending, while tackling the issue of the â€œDreamers,â€ people brought illegally to the country as children. Trump in September put a March 2018 expiration date on the Deferred Action for Childhood Arrivals, or DACA, program, which protects the young immigrants from deportation and provides them with work permits. The president has said in recent Twitter messages he wants funding for his proposed Mexican border wall and other immigration law changes in exchange for agreeing to help the Dreamers. Representative Debbie Dingell told CBS she did not favor linking that issue to other policy objectives, such as wall funding. â€œWe need to do DACA clean,â€ she said.  On Wednesday, Trump aides will meet with congressional leaders to discuss those issues. That will be followed by a weekend of strategy sessions for Trump and Republican leaders on Jan. 6 and 7, the White House said. Trump was also scheduled to meet on Sunday with Florida Republican Governor Rick Scott, who wants more emergency aid. The House has passed an $81 billion aid package after hurricanes in Florida, Texas and Puerto Rico, and wildfires in California. The package far exceeded the $44 billion requested by the Trump administration. The Senate has not yet voted on the aid.  As U.S. budget fight looms, Republicans flip their fiscal script"
        vectorLSTM = tokenizer.texts_to_sequences([inputText])
        maxlen = 700
        vectorLSTM = pad_sequences(vectorLSTM,maxlen=maxlen)
        
        vectorTFIDF = tfidfVectorizer.transform([inputText])
        vectorBOW = countVectorizer.transform([inputText])
        
        #print(RF_TFIDF.predict(vectorTFIDF))
        #print(RF_BOW.predict(vectorBOW))
        #print(SVM_TFIDF.predict(vectorTFIDF))
        #print(SVM_BOW.predict(vectorBOW))
        #print(NB_TFIDF.predict(vectorTFIDF))
        #print(NB_BOW.predict(vectorBOW))
        #print(model.predict(vectorLSTM))
        
        classifiers = ["RF_TFIDF","RF_BOW","SVM_TFIDF","SVM_BOW","NB_TFIDF","NB_BOW","LSTM"]
        results = [RF_TFIDF.predict_proba(vectorTFIDF)[0][1],RF_BOW.predict_proba(vectorBOW)[0][1],SVM_TFIDF.predict(vectorTFIDF)[0],SVM_BOW.predict(vectorBOW)[0],NB_TFIDF.predict(vectorTFIDF)[0],NB_TFIDF.predict(vectorBOW)[0],model.predict(vectorLSTM)[0][0]]
        
        sourcedata = pd.DataFrame({'Classifiers':classifiers,'Detection':results})
        
        st.write(alt.Chart(sourcedata).mark_bar().encode(x='Classifiers',y='Detection', tooltip='Detection').properties(
    width=400,
    height=600
))
        
        #
        
    elif(selected_page == "Analysis"):
        print("ololo")
    elif(selected_page == "Metrics"):
        print("ololo")
    st.write(""" 
    ## Made By : """)
    sunset_imgs = [
    'Nicole.jpeg',
    'Ola.jpeg',
    'Trivikram.jpg',
    
    ]
    image_iterator = paginator("", sunset_imgs)
    indices_on_page, images_on_page = map(list, zip(*image_iterator))
    indices_on_page[0] = "Nicole"
    indices_on_page[1] = "Ola"
    indices_on_page[2] = "Trivikram"
    
    
    st.image(images_on_page, width=100, caption=indices_on_page)
if(__name__ == "__main__"):
    main()

