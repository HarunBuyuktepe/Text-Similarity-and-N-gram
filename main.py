# importing the required modules
import PyPDF2
from os import path
import mammoth
import string
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import operator
from wordcloud import WordCloud
from pandas import DataFrame
import matplotlib.pyplot as plt
import csv
import numpy as np
import os



def readFile(p,i):
    mergedLines = []

    if(i.endswith(".pdf")):
        # creating a pdf File object of original pdf
        pdfFileObj = open(path.join(p,i), 'rb')

        # creating a pdf Reader object
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

        for page in range(pdfReader.numPages):# add all page text to merged text
            # creating page object
            pageObj = pdfReader.getPage(page)
            # add pdf's text to text
            text = pageObj.extractText();
            # add lines
            lines = [text]
            lines = re.sub("[^\w]", " ", lines[0]).split() # change unnecessary symbols to null
            mergedLines = mergedLines + lines
        pdfFileObj.close()
    elif(i.endswith(".txt")):
        txtFile = open(path.join(p, i), "r")
        lines = txtFile.readlines()
        lines = re.sub("[^\w]", " ", lines[0]).split() # change unnecessary symbols to null
        mergedLines = mergedLines + lines
    elif(i.endswith(".docx")):
        with open(path.join(p, i), "rb") as docx_file:
            result = mammoth.extract_raw_text(docx_file)
            text = result.value
            lines = [text]
            lines = re.sub("[^\w]", " ", lines[0]).split() # change unnecessary symbols to null
            mergedLines = mergedLines + lines

    mergedLines = [i.lower() for i in mergedLines]  # Convert all characters to lowercase before tokenizing.
    mergedLines = [''.join(c for c in s if c not in string.punctuation) for s in mergedLines] # Tokenizing our text
    mergedLines = [i for i in mergedLines if i not in stop] # Add all word if it is not stop word.
    #print(mergedLines)
    return mergedLines # return processed text



# Defining stop words
stop = set(stopwords.words('english'))


def makeDataframeAndWriteToCsv(freq,vocab,columnName,filename):
    df=DataFrame(freq,vocab) # set matrix view to write csv file
    df.columns=[columnName] # column name defining
    df=df.sort_values(by=columnName,ascending=0) # sort ascending order
    df=df.nlargest(50,columnName) # elect 50 words which repeated more

    df.to_csv(filename) # write selected word with fitting form


def makeWordCloud(filename):
    reader_list = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        reader_list = '\t'.join([i[0] for i in reader])

    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(reader_list)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    wordcloud.to_file("images/"+filename+".png")


def main():
    allword=[]; # Our file pool
    # Defining files' path
    dire = path.dirname(__file__)
    # Defining files' direction to path
    p = os.path.join(dire, 'file/')
    # calling the file opener and processor for all element of file
    for i in os.listdir(os.path.join(dire, 'file')):
        # Give processed text
        mergedLines=readFile(p,i) # p: path; i: file name;
        allword+=mergedLines # add all elected word to pool
        vectorizer = CountVectorizer() #
        X = vectorizer.fit_transform(mergedLines) # return term-document matrix which word is in which line
        freq = np.ravel(X.sum(axis=0))  # sum along axis 0 on term-document matrix
        vocab = [v[0] for v in sorted(vectorizer.vocabulary_.items(), key=operator.itemgetter(1))];
        # create term frequency files
        makeDataframeAndWriteToCsv(freq, vocab, 'tf values', 'tf_list'+i+'.csv')
        makeWordCloud('tf_list'+i+'.csv')


    # get tfidf values
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(allword)
    idf = vectorizer.idf_ # idf vector

    # duties that write in project document
    makeDataframeAndWriteToCsv(idf, vectorizer.get_feature_names(), 'tf-idf values', 'tfidf_list.csv')
    makeWordCloud('tfidf_list.csv')



if __name__ == "__main__":
    # calling the main function
    main()

