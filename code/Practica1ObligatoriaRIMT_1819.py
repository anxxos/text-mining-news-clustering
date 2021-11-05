
"""
Práctica Final
Recuperación de Información y Minería de Textos
Máster en Data Science
ETSII-URJC
"""

import re, os, numpy
import nltk, spacy, string

from bs4 import BeautifulSoup
#from sklearn.metrics.cluster import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
# from tikapp import TikaApp
from collections import Counter
from textblob import TextBlob

class TextProcessor:
    def __init__(self, text):
        """
        Constructor for processing texts.

        Parameters
        ----------
        text: text to analyse.
        """
        self.text = text
        
    def removeSymbols(self):
        """ Removing punctuation and symbols."""
        
        # Initial variable declaration:
        text = self.text
        words = []
        punctuation = list(string.punctuation)
        pos_not_wanted = ['PUNCT','SPACE']

        # Translating some symbols.
        text = re.sub('\$', " dollar ", text)
        text = re.sub('\%', " percent ", text)
        
        # Dividing words with a dot or quotation marks
        # in between them:
        #text = text.split('.')
        #text = ' '.join(text)
        #text = text.split('”')
        #text = ' '.join(text)
        
        # Transforming to spacy.doc.Doc object for easier processing.   
        doc = nlp(text)

        # Removing punctuation 
        # (double way with pos_tags and list of punctuation),
        # and the rest of symbols and also numbers. 
        
        for word in [token for token in doc 
                     if token.pos_ not in pos_not_wanted
                     and re.search('[a-zA-Z]', token.text)]:
            if word.text not in punctuation:
                words.append(word.text)

        text = ' '.join(words)
        return text
        
    def removeStopWords(self, nouns):
        """ Removing stopwords and more non-useful words.
        Parameters
        ----------
        nouns: bool
            If True, removes all words but noun words.
        """      
        
        text = self.text
        stopwords_en = nltk.corpus.stopwords.words("english")
        pos_not_wanted = ['DET','SPACE','ADV','ADP','PART']
        words = []
        
        # Transforming to spacy.doc.Doc object for easier processing.   
        doc = nlp(text)
        
        # Double removing stop_words with Spacy and NLTK.
        tokens = [token for token in doc 
                  if token.pos_ not in pos_not_wanted
                  and not token.is_stop]
        if nouns == True:
            tokens = [token for token in tokens if token.pos_ == 'NOUN']
        
        lengths = [len(token) for token in tokens]
        
        for word in tokens:
            if word.text not in stopwords_en:                
                # Removing outliers in word-length:
                if (len(word.text) >= numpy.percentile(lengths,5)) &\
                   (len(word.text) <= numpy.percentile(lengths,95)):
                       words.append(word.text)                
                
        text = ' '.join(words)       
        return text
    
    def rootText(self, lem = True, stem = False):
        """ Perform lemmatization and stemming.
        
        Parameters
        ----------
        lem: bool
            True by default. Returms token lemma.
        stem: bool
            False by default. Returns token stem.
            
        """
        text = self.text
        words = []
        
        # Transforming to spacy.doc.Doc object for easier processing.   
        doc = nlp(text)
        
        # Performing lematization and/or stemming to non-proper nouns:
        if lem == True:
            words = [str(token.lemma_) for token in doc if not token.pos_ == 'PROPN']
            words.extend([token.text for token in doc if token.pos_ == 'PROPN'])
            if stem == True:
                raise ValueError("Arguments cannot be True nor False simultaneously.")
            elif stem == False:
                return ' '.join(words) 
        else:
            if stem == True:
                stemming = nltk.SnowballStemmer('english')
                words = [stemming.stem(word.text) for word in 
                         [token for token in doc if not token.pos_ == 'PROPN']]
                words.extend([token.text for token in doc if token.pos_ == 'PROPN'])
                return ' '.join(words) 
            else:
                raise ValueError("Arguments cannot be True nor False simultaneously.")
                
    def getNamedEntities(self, filtering):
        """ NER """
        text = self.text
        doc = nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ in filtering]
    
class TextClassifier:
    def __init__(self, terms, docs):
        """
        Parameters
        ----------
        docs: list of spacy.doc.Doc objects
        terms: list of str
        """
        self.docs = docs
        self.terms = terms
        
    def TFIDF(self, doc, idf = False, tf = True):
        """
        Term Frequency–Inverse Document Frequency, numerical statistic 
        intended to reflect how important a word is to a document 
        in a collection.
        
        Parameters
        ----------
        doc: spacy.doc.Doc
        tf: bool
            Default value is True.
        idf: bool
            Default value is False.
            
        Reference
        ---------
        For the calculus of the tf and idf terms, the following code,
        from the NLTK library, has been adapted:
            
        Natural Language Toolkit: Texts
        Copyright (C) 2001-2018 NLTK Project
        """
        terms = self.terms
        docs = self.docs
        word_tf = []

        for term in terms:
            if (idf == False) and (tf == True):
                # The frequency of the term in text.
                tf_ = doc.text.count(term)/len(doc)
                word_tf.append(tf_)
            elif (idf == True) and (tf == False):
                # The inverse document frequency.
                matches = len([True for doc in docs if term in doc.text])
                idf_ = (numpy.log10(len(docs) / matches) if matches else 0.0)
                word_tf.append(idf_)
            elif (idf == True) and (tf == True):
                # TF & IDF.
                tf_ = doc.text.count(term)/len(doc)
                matches = len([True for doc in docs if term in doc.text])
                idf_ = (numpy.log10(len(docs) / matches) if matches else 0.0)
                
                # Dot product of tf and idf.
                word_tf.append(tf_*idf_) 
            else:
                raise ValueError('At least one argument must be True.')
        return word_tf
    
    def vectorText(self):
        """ Transform the TF-IDF into document vectors. """
        docs = self.docs
        
        # And here we actually call the function and create our array of vectors.
        vectors = numpy.array([self.TFIDF(doc, idf=True) for doc in docs])

        print("Vectors created.")
        return vectors
        
    def clusterTexts(self, clustersNumber, distance):
        """
        Method for classifying the array of vectors generated in the constructor
        with an Agglomerative Clustering algorithm.

        Parameters
        ----------
        clustersNumber: int
            Number of clusters for the algorithm to make.
        distance: str
            Distance type for calculations. Values within 'cosine' and 'euclidean'.
        """
        vectors = self.vectorText()
        #print(vectors)
        
        # Initializing the clusterer:
        clusterer = AgglomerativeClustering(n_clusters=clustersNumber,
                                          linkage="average", affinity=distanceFunction)
        clusters = clusterer.fit_predict(vectors) 
    
        return clusters

# Deprecated. From a previous version of this script.
#class TikaReader:
#    """
#    A Tika class for performing tasks such as detecting languages
#    or translating documents.
#    """
#    # Constructor:
#    def __init__(self, file_process):
#        # Tika Python Client:
#        self.tika_client = TikaApp(file_jar="tika-app-1.20.jar")
#        self.file_process = file_process
#
#    # Document language detector:
#    def detect_language(self):
#        return self.tika_client.detect_language(self.file_process)
    
def extractText(folder, listing):
    """
    Function for extracting text from HTML news articles in folder
    within the same parent directory.
    
    Parameters
    ----------
    folder: str
        Folder in which the documents are.
    listing: list
        List of documents.
    """
    if file.endswith(".html"):
        url = folder+"/"+file
    try:
        f = open(url,encoding="utf-8")
        raw = f.read()
    except UnicodeDecodeError:
        f = open(url,encoding="latin-1")
        raw = f.read()
    f.close()
    soup = BeautifulSoup(raw,'html.parser')
    text = ""

    for node in soup.find_all('p',itemprop='articleBody'):
        text = text + node.text
    if text == '':
        for node in soup.find_all('p',class_='article-body'):
            text = text + node.text
    if text == '':
        for node in soup.find_all('p'):
            text = text + node.text
    
    text = text.replace('\n','')
    return text

def translateText(text, file, folder, format_file, from_lang, to_lang):
    """
    Function for translating and saving text to given format and folder
    within the same parent directory.
    
    Parameters
    ----------
    text: str
        Text for translating.
    file: str
        Name of the document.
    folder: str
        Desired folder in which to save the newly translated document.
    format_file: str
        Desired format.
    from_lang: str
        Original language.
    to_lang: str
        Desired language.
    """
    # Detecting and translating language with TextBlob:
    text = TextBlob(text)

    if text.detect_language() == from_lang:
        text = text.translate(from_lang=from_lang, to=to_lang)
    text = str(text)
    
    # Removing non-ascii characters.
    text = re.sub('[^\x00-\x7F]+', '', text)
    
    # Saving text:
    filename = (folder, file.split('.')[0], format_file)
    with open("./{folder}/{name}.{format_file}".format(folder=filename[0],\
              name=filename[1],format_file=filename[2]), "w") as file:
        file.write(text) 
    file.close() 
    
    return text

def readTXT(file, folder):
    """ 
    Function for reading txt files in a folder within 
    the same parent directory.
    
    Parameters
    ----------
    file: str
        Name of the document.
    folder: str
        Document folder.
    """
    f = open(folder+'/'+file,'r')
    text = f.read()
    
    return text

if __name__ == "__main__":
    
    # Folder in which the documents are:
    # Previous version.
    folder_html = "CorpusHTMLNoticiasPractica1819"
    
    # Current version.
    folder_txt = "CorpusTXTNoticiasPractica1819" 
    
    # Memory reservation:
    or_texts, or_docs, docs, ents, terms, nouns = [], [], [], [], [], []
    langs_before, langs_after = [], []
    
    # NLP model calling:
    nlp = spacy.load('en')

    # Documents processing:
    listing_html = sorted(os.listdir(folder_html))
    listing_txt = sorted(os.listdir(folder_txt))

    if os.listdir(folder_txt) == []:
        for file in listing_html:
            
            # Extracting text from HTML documents.
            # Previous version.
            text = extractText(folder_html,listing_html)
            
            # Detecting language with Tika.
            # Deprecated. Outperformed by TextBlob.
            #processor = TikaReader(folder+'/'+file)
            #lang = processor.detect_language()
            #langs_before.append(lang)
            
            # Detecting language with TextBlob:
            langs_before.append(TextBlob(text).detect_language())   
            
            # Translating and saving to TXT files.
            text = translateText(text,file,folder_txt,'txt','es','en')
        
    for file in listing_txt:
        
        # Reading previously saved TXT document files.
        text = readTXT(file, folder_txt)
        
        # Detecting language with TextBlob 
        # (just checking everything is translated):
        langs_after.append(TextBlob(text).detect_language())           
        
        # Appending list of original texts:
        or_texts.append(text)
        
        # Creating objects of type spacy.tokens.doc.Doc
        # with original texts:
        doc = nlp(text)
        or_docs.append(doc)
        
        # Processing text:
        text = TextProcessor(text).removeSymbols()
        text = TextProcessor(text).removeStopWords(nouns = False)
        #text = TextProcessor(text).rootText(lem = True, stem = False)
        doc = nlp(text)
        docs.append(doc)
        
        # Getting named entities alone:
        ents_doc = TextProcessor(text).getNamedEntities(['PERSON'])
        ents.extend(ents_doc)
        
        # Getting the unique terms:
        """ 
        Source
        ------
        For the calculus of spaCy unique terms, the 
        following code has been used:
            
        Mathew Honnibal on Oct 14, 2015
        https://github.com/explosion/spaCy/issues/139
        """
        counts = doc.count_by(spacy.attrs.ORTH)
        for word_id, count in sorted(counts.items(), reverse=True,\
                                     key=lambda item: item[1]):
            terms.append(nlp.vocab.strings[word_id])
           
    # End of for loop.
    terms = list(set(terms))
    ents = list(set(ents)) # unique entities

    print("Prepared ", len(docs), " documents.")
    print("They can be accessed using or_texts[n], being n an integer from 0 to "\
          + str(len(docs)-1) + ".")
    if langs_before != []:
        print("Distribution of documents by language before translation: ",\
              dict(Counter(langs_before)))
    print("Distribution of documents by language after translation: ",\
          dict(Counter(langs_after)))
    print("Unique terms found: ", len(terms))
    print("Named entities found: ", len(ents))
    
    # Selecting the distance function for calculating distances between 
    # the document vectors subsequently calculated:
    #distanceFunction = "euclidean"
    distanceFunction = "cosine"
    
    # Applying the classifier (TF-IDF, vectorizing and clustering):
    test = TextClassifier(ents, docs).clusterTexts(6, distanceFunction)
    print("Test: ", test)
    
    # Gold Standard pattern:
    # Old.
    #reference =[0, 1, 2, 2, 3, 2, 2, 2, 4, 0, 0, 3, 3, 4, 2, 3, 0, 4, 4, 4, 5, 2]
    # New.
    #reference =[0, 1, 2, 2, 2, 3, 2, 2, 2, 4, 0, 0, 3, 3, 4, 2, 3, 0, 4, 4, 5, 1]
    # Newest new.
    reference =[0, 5, 2, 2, 2, 3, 2, 2, 2, 4, 0, 0, 3, 3, 4, 2, 3, 0, 4, 4, 4, 1]
    print("Reference: ", reference)

    # Evaluation with ARI:
    print("Adjusted Rand Index: ", round(adjusted_rand_score(reference,test),8))