import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

mssgs = pd.read_csv('data-sets/SMSSpamCollection',sep='\t',header=None,names =['label','sms'])

mssgs.info()

# []
#percentage of spams vs. non-spam
mssgs.value_counts(['label'],normalize=True)
mssgs.value_counts(['label'],normalize=True).plot(kind='barh')

#a good idea would be to create a separate training set of about 80% of the data
#the remaining 20% will be the test set to test our spam filter
eighty = mssgs.shape[0]*.80
#eighty will be rounded to 4458

# frac=1 used to randomize the dataset before slicing
mssgs = mssgs.sample(frac=1,random_state=1)
training_set = mssgs.copy().iloc[0:4458,:]
test_set = mssgs.copy().iloc[4458:,:]

# []
#will reset indexes to prevent any problems in future
training_set.reset_index(inplace=True,drop=True)
test_set.reset_index(inplace=True,drop=True)

# now going to break apart the messages and create columns
# which express the frequencies of each word without punctuation

training_set.iloc[-1,:].str.replace('[\W]',' ')
#example of process
training_set['sms'] = training_set.copy()['sms'].str.replace('[\W]',' ')
training_set['sms'] = training_set['sms'].copy().str.lower()

# create a list with all of the unique words that occur
training_set['sms'] = training_set.copy()['sms'].str.split()

vocabulary = []
for row in training_set['sms']:
    for a in row:
        vocabulary.append(a)

# transform to set
len(vocabulary)
vocabulary = set(vocabulary)
len(vocabulary)
vocabulary = list(vocabulary)

# find the frequencies of each word in each mssg

    #make each word a column



#some.rename({'sms':'sms2'},axis=1) #believed to be duplicate

df_1 = pd.DataFrame(columns=vocabulary)

# practice
# some = training_set.iloc[:5,:]
# some = some.copy()
# some2 = some.rename({'sms':'sms2'},axis=1).copy()
# df_2 = pd.concat([some2,df_1],axis=1)
# import numpy as np
# df_2 = df_2.copy()
# cols = df_2.columns[2:]


# df_2 = df_2.copy()


# for col in cols :
#     for val,row in enumerate(df_2['sms2']) :

#         for word in row:
#             if word == col :
#                 if np.isnan(df_2.iloc[val,:][col]):
#                     df_2.iloc[val,:][col] = 1
#                 else :
#                     df_2.iloc[val,:][col] += 1



# doesnt work making changes on slice

# # for each row if word is equal to column add a one
# import copy

# training_set2 = training_set2.copy()
# cols = df_1.columns[2:]
# import numpy as np
# for val2,col in enumerate(cols) :
#     for val,row in enumerate(training_set2['sms2']) :
        
#         for word in row:
#             if word == col :
                
#                 if np.isnan(training_set2.iloc[val,:][col]):
#                     training_set2.iloc[val,:][col] = 1
#                 else :
#                     training_set2.iloc[val,:][col] += 1

#[]
training_set = training_set.copy()
training_set = training_set.rename({'sms':'sms2'},axis=1).copy()
training_set2 = pd.concat([training_set,df_1],axis=1)
cols = training_set2.columns[2:]
dicts = [] # this will be a list of dictionaries
for row in training_set2['sms2']: # iterating through each list of words in mssg
    dict1= {} # unique dictionary for each mssg
    for col in cols:
        dict1[col] = 0 # making each column a key in the dictionary
        for word in row :
            if word == col :
                dict1[col] += 1 # add one each time a word matches the column

    dicts.append(dict1)

training_set3 = pd.DataFrame(dicts)

training_set2_0 = training_set2.iloc[:,0:2].copy()
training_set4 = training_set2_0.merge(training_set3,left_index=True,right_index=True)

#[]
#dq method

word_counts_per_sms = {key : [0] * training_set.shape[0] for key in vocabulary}

for index,row in enumerate(training_set['sms2']) :
    for word in row :
        word_counts_per_sms[word][index] += 1

t_set = pd.DataFrame(word_counts_per_sms)
t_set2 = pd.concat([training_set,t_set],axis=1)

# []
# given our naive bayes formula we will calculate the constants
p_spam = t_set2['label'].value_counts(normalize=True)['spam']
p_spamc = t_set2['label'].value_counts(normalize=True)['ham']
n_of_vocabulary = len(vocabulary) #already obtained
N_spam = t_set2.loc[t_set2['label'] == 'spam','sms2'].apply(len).sum()
N_spamc = t_set2.loc[t_set2['label'] == 'ham','sms2'].apply(len).sum()
alpha = 1 # for Laplace smoothing

#[]
#Now the parameters will be calculated fortunately the calculations
    #can be done beforehand these calculations will be placed
    # in a dictionary

# for each wordfind P(w|Spam) & P(w|Spamc)

#initialize dict
wordp_dict = {}

#loop each word in vocabulary
for word in vocabulary:
# for spam mssgs calculate prob for word
    freq_given_spam = t_set2.loc[t_set2['label'] == 'spam',word].sum()
    p_word_given_spam = (freq_given_spam + alpha) / (N_spam +
        alpha * n_of_vocabulary)
    key_spam = '{}_spam'.format(word)
    wordp_dict[key_spam] = p_word_given_spam
# for spam mssgs calculate prob for word
    freq_given_spamc = t_set2.loc[t_set2['label'] == 'ham',word].sum()
    p_word_given_spamc = (freq_given_spamc + alpha) / (N_spamc +
        alpha * n_of_vocabulary)
    key_spamc = '{}_spamc'.format(word)
    wordp_dict[key_spamc] = p_word_given_spamc

#[]
#dq method
wordp_spam = {}
wordp_spamc = {}

spam_df = t_set2[t_set2['label'] == 'spam']
spamc_df = t_set2[t_set2['label'] == 'ham']
for word in vocabulary:
    freq_given_spam1 = spam_df[word].sum()
    p_word_given_spam1 = (freq_given_spam1 + alpha) / (N_spam +
        alpha * n_of_vocabulary)
    key_spam = '{}_spam'.format(word) #omitted
    wordp_spam[word] = p_word_given_spam1

    #spamc
    freq_given_spamc1 = spamc_df[word].sum()
    p_word_given_spamc1 = (freq_given_spamc1 + alpha) / (N_spamc +
        alpha * n_of_vocabulary)
    key_spamc = '{}_spamc'.format(word) #omitted
    wordp_spamc[word] = p_word_given_spamc1

#[]
#creating a new function to go process algo on the sentence
import re
def classify(message):
    message = re.sub(r'\W',repl = ' ',string=message)
    message = message.lower()
    message = message.split() # issues arise when using str.split()




    p_spam_given_message = p_spam
    p_spamc_given_message = p_spamc

    for word in message:
        if word in wordp_spam:
            p_spam_given_message *= wordp_spam[word]

        if word in wordp_spamc:
            p_spamc_given_message *= wordp_spamc[word]


    if p_spamc_given_message > p_spam_given_message:
        return 'ham'
    elif p_spamc_given_message < p_spam_given_message:
        return 'spam'
    else:
        return 'human verification'

#[] quick tests
spam_ex = 'WINNER!! This is the secret code to unlock the money: C3421.'

spamc_ex = 'mom it\'s raining can you come scoop me please?' #wasn't considered spam

equal_ex = 'vale sobretodo por aqui en plan algo de locales te recomiendo que vayas por el centro' #wasn't considered spam

for a in [spam_ex,spamc_ex,equal_ex] :
    classify(a)

#[]
test_set['tested'] = test_set['sms'].apply(classify)

accuracy = (test_set['label'] == test_set['tested']).sum() / test_set.shape[0]

'''Next steps'''
# print the correct vs incorrect spam designations
#Isolate the 14 messages that were classified incorrectly and try to figure out why the algorithm reached the wrong conclusions.
#Make the filtering process more complex by making the algorithm sensitive to letter case.


