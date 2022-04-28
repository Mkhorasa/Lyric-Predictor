import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

lyrics = input('Welcome to Lyrics classifing machine by Media. please give me text:' )

Filename1= './week4/finalized_vectorizer.sav'
vc=pickle.load(open(Filename1, 'rb'))

filename = './week4/finalized_model.sav'
rf=pickle.load(open(filename, 'rb'))

lyrics_v = vc.transform([lyrics])

artist_prediction = rf.predict(lyrics_v)

if artist_prediction=='Britney spears':
    print('This Text belongs to Britney spears')
if artist_prediction=='50 Cent':
    print('This Text belongs 50 Cent')