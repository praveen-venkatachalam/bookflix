
from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn import svm

app = Flask(__name__)

final_rating = pd.read_csv("book_pivot.csv", index_col=0)
book_pivot = final_rating.pivot_table(columns='user_id', index='title', values="rating")
book_pivot.fillna(0, inplace=True)

model=pickle.load(open('nnmodel.pkl','rb'))

# query_index = np.random.choice(book_pivot.shape[0])



@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/pred')
def hell_world():
    return render_template("pred.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    book_dict = dict(enumerate(book_pivot.index))
    def get_book_index(book_dict, bookname):
        for index, title in book_dict.items():
            if title.upper() == bookname.upper():
                return index
        return None
    
    y = []
    for x in request.form.values():
        y.append(x)
    book_index = get_book_index(book_dict, y[0])
    print(book_index)
    query_index = book_index

    distances, indices = model.kneighbors(book_pivot.iloc[query_index, :].values.reshape(1, -1), n_neighbors = 6)
    arr = []
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print('Recommendations for {0}:\n'.format(book_pivot.index[query_index]))
        else:
            print('{0}: {1}, with distance of {2}:'.format(i, book_pivot.index[indices.flatten()[i]], distances.flatten()[i]))
            arr.append(book_pivot.index[indices.flatten()[i]])

    return render_template('index.html',pred=arr)


if __name__ == '__main__':
    app.run(debug=True)