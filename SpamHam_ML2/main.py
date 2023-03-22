import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt
import seaborn as sns

import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

def rmv_spec_char(body):
    """Remove special characters.

    Args:
        body (list): Raw Text Body.

    Returns:
        list: Clean Text Body
    """
    body_clean = []
    for char in body:
        if char.isalnum():
            body_clean.append(char)
    return body_clean

def rmv_sw_pm(body):
    """Remove punctuation marks and stop words.

    Args:
        body (list): Raw Text Body.

    Returns:
    """    
    body_clean = []
    for char in body:
        if char not in stopwords.words('english') and char not in string.punctuation:
            body_clean.append(char)   
    return body_clean

def stemm_text(body):
    """Stemm the words, e.g. walk -> walk, walked -> walk, walks -> walk. etc.

    Args:
        body (list): Raw Text Body.

    Returns:
    """  
    body_clean = []
    stemmer = SnowballStemmer('english')
    for char in body:
        body_clean.append(stemmer.stem(char))
    
    return body_clean
    

def prepro_text_body(body):
    body = body.lower()
    body = nltk.word_tokenize(body)
    body = rmv_spec_char(body)
    body = rmv_sw_pm(body)
    body = stemm_text(body)
    print("done")
    return " ".join(body)

def get_word_list(df, type = "ham"):
    word_list = []
    if type == "ham":
        id = 0
    elif type == "spam":
        id = 1
    for body in df[df['Label'] == id]['Body_clean'].tolist():
        for word in body.split():
            word_list.append(word)
    return word_list

def get_vocab_distribution(mail_df, vocabulary):
    word_counts_per_mail = {unique_word: [0] * len(mail_df['Body_clean']) for unique_word in vocabulary}

    for index, sms in enumerate(mail_df['Body_clean']):
        for word in sms:
            word_counts_per_mail[word][index] += 1
            
    word_counts = pd.DataFrame(word_counts_per_mail) 
    return word_counts, word_counts_per_mail

def calc_parameters():
    pass


def classify_msg(msg,p_spam, p_ham, parameters_spam, parameters_ham):
    msg = prepro_text_body(msg)
    msg = msg.split()
    
    p_spam_ = p_spam
    p_ham_ = p_ham
    
    for word in msg:
        if word in parameters_spam:
            p_spam_ *= parameters_spam[word]

        if word in parameters_ham: 
            p_ham_ *= parameters_ham[word]

    #HAM
    if p_spam_ <= p_ham_:
        return p_spam_, p_ham_, 0
    #SPAM
    else:       
        return p_spam_, p_ham_, 1

def main():
    path = "Data/ML2/completeSpamAssassin.csv"
    mail_df = pd.read_csv(path)
    
    #https://www.kaggle.com/code/rohitshirudkar/email-classification-spam-or-ham
    
    print(mail_df)
    
    #cleaning up data-set
    mail_df.dropna(inplace=True)
    mail_df.drop(['Unnamed: 0'],axis=1, inplace=True)
    
    #adding base features
    mail_df['no_char'] = mail_df['Body'].apply(len)
    mail_df['no_words'] = mail_df['Body'].apply(lambda x:len(nltk.word_tokenize(x)))
    mail_df['no_sentences'] = mail_df['Body'].apply(lambda x:len(nltk.sent_tokenize(x)))
    
    #preprocessing text body
    if False:
        mail_df['Body_clean'] = mail_df['Body'].apply(prepro_text_body)
    
    mail_df = pd.read_pickle("Data/ML2/SpamHamClean.pkl")
    
    #reduce size to 50 mails
    mail_train = mail_df.sample(n=1000, random_state = 1)
    
    #ham word list
    ham_list = get_word_list(mail_train, "ham")
    
    #spam word list
    spam_list = get_word_list(mail_train, "spam")
    
    #split the string into list of strings
    mail_train['Body_clean'] = mail_train['Body_clean'].str.split()
    
    #create vocabulary of unique words
    vocabulary = list(set(ham_list + spam_list))

    word_counts, word_counts_per_mail = get_vocab_distribution(mail_train, vocabulary)
    mail_train = mail_train.reset_index(drop=True)
    mail_df_clean = pd.concat([mail_train, word_counts], axis=1)


    # Isolating spam and ham messages first. 1 == spam, 0 == ham
    spam_messages = mail_df_clean[mail_df_clean['Label'] == 1]
    ham_messages = mail_df_clean[mail_df_clean['Label'] == 0]

    # P(Spam) and P(Ham)
    p_spam = len(spam_messages) / len(mail_df_clean)
    p_ham = len(ham_messages) / len(mail_df_clean)

    # N_Spam
    n_words_per_spam_message = spam_messages['Body_clean'].apply(len)
    n_spam = n_words_per_spam_message.sum()

    # N_Ham
    n_words_per_ham_message = ham_messages['Body_clean'].apply(len)
    n_ham = n_words_per_ham_message.sum()

    # N_Vocabulary
    n_vocabulary = len(vocabulary)

    # Laplace smoothing
    alpha = 1
    #vectorizer
    
    parameters_spam = {unique_word:0 for unique_word in vocabulary}
    parameters_ham = {unique_word:0 for unique_word in vocabulary}

    # Calculate parameters
    for word in vocabulary:
        n_word_given_spam = spam_messages[word].sum() # spam_messages already defined
        p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha*n_vocabulary)
        parameters_spam[word] = p_word_given_spam

        n_word_given_ham = ham_messages[word].sum() # ham_messages already defined
        p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + alpha*n_vocabulary)
        parameters_ham[word] = p_word_given_ham
    
    no_test_mails = 50
    mail_test =  mail_df.sample(n=no_test_mails, random_state = 1)
    
    correct = 0
    total = 0
    tp, tn, fp, fn = 0,0,0,0
    for index, mail in mail_test[['Body','Body_clean','Label']].iterrows():
        print(mail['Body'])   
        p_spam_, p_ham_, pred = classify_msg(mail['Body_clean'],p_spam, p_ham, parameters_spam, parameters_ham)
        
        if pred == 1:
            print(f"SPAM DETECTED: {p_spam_} | {p_ham_}")
            if mail['Label'] == 1:
                print("CORRECT!")
                tp += 1
                correct = correct + 1
            else:
                print("WRONG!")
                fp += 1
        else:
            print(f"HAM DETECTED: {p_spam_} | {p_ham_}")
            if mail['Label'] == 1:
                print("WRONG!")
                fn += 1
            else:
                print("CORRECT!")
                correct = correct + 1
                tn += 1
        total = index
    
    print(f"{correct} of {no_test_mails} were labeled correctly, this is {correct / no_test_mails * 100}%")
    print(f"TP = {tp} || TN = {tn} || FP = {fp} || FN = {fn}")
    print(f"TPR / Sensitivity = {tp/(tp+fn)}")
    print(f"TNR / Specificity = {tn/(tn+fp)}")
    
    
    #Test     
    msg_ham = "Hello, are you up for dinner later? I would like to discuss the situation with our new supplier."    
    (p_spam1, p_ham1, prediction1) = classify_msg(msg_ham,p_spam, p_ham, parameters_spam, parameters_ham)
    
    msg_spam = "HELO SPICY PENIS FRIEND, LOVE YOU LONG TIME!!!!!! MORE DRUGS LESS SAD ALWAYS HAPPY"    
    (p_spam2, p_ham2, prediction2) = classify_msg(msg_spam,p_spam, p_ham, parameters_spam, parameters_ham)


if __name__ == "__main__":
    main()