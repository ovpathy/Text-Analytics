{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 163,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from matplotlib.colors import ListedColormap\n",
    "from sklearn import neighbors, datasets, preprocessing, tree, model_selection,svm\n",
    "from sklearn.feature_selection import SelectFromModel,RFECV\n",
    "from sklearn.linear_model import LogisticRegression, LinearRegression\n",
    "from sklearn.svm import SVR, LinearSVC\n",
    "from sklearn.ensemble import ExtraTreesClassifier\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict, StratifiedKFold\n",
    "import sklearn.metrics as metrics\n",
    "from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, roc_curve, auc, mean_absolute_error\n",
    "from scipy import interp\n",
    "from ggplot import *\n",
    "import re\n",
    "from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.tokenize import RegexpTokenizer\n",
    "import nltk\n",
    "lemma = nltk.wordnet.WordNetLemmatizer()\n",
    "from nltk.stem.snowball import SnowballStemmer\n",
    "sb = SnowballStemmer(\"english\")\n",
    "from nltk.stem.porter import *\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "stemmer = PorterStemmer()\n",
    "from sklearn.ensemble import GradientBoostingRegressor as GBR\n",
    "from sklearn.ensemble import AdaBoostRegressor as ABR\n",
    "from sklearn.neural_network import MLPRegressor as NN\n",
    "from sklearn.neighbors import KNeighborsRegressor \n",
    "from statsmodels.tools.eval_measures import rmse"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 168,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "def calculate_mape(y_true,y_pred):\n",
    "    mape=np.mean(np.abs((y_true - y_pred) / y_true)) *100\n",
    "    # rmse(y_true,y_pred)\n",
    "    return mape\n",
    "\n",
    "def to_tfidf(df):\n",
    "    df['snippet']=df['snippet'].apply(lambda words: tokenizer.tokenize(words))\n",
    "    df['snippet']=df['snippet'].apply(lambda word: [sb.stem(item) for item in word if item not in stop])\n",
    "    df['snippet']=df['snippet'].apply(lambda word: ' '.join(word))\n",
    "    X_train_tfidf = TfidfVectorizer(max_df=0.8, min_df=0.05).fit_transform(df['snippet'])\n",
    "    actual=df['twitter']\n",
    "    return X_train_tfidf,actual\n",
    "   \n",
    "def run_svr(predictors,target):\n",
    "    #rtree = DecisionTreeRegressor(max_depth=25)\n",
    "    rtree = SVR()\n",
    "    predicted_rtree=cross_val_predict (rtree, predictors,target, cv=10)\n",
    "    mape_rtree=calculate_mape(target,predicted_rtree)\n",
    "    return mape_rtree\n",
    "\n",
    "def run_nn(predictors,target):\n",
    "    #rtree = DecisionTreeRegressor(max_depth=25)\n",
    "    rtree = NN()\n",
    "    predicted_rtree=cross_val_predict (rtree, predictors,target, cv=10)\n",
    "    mape_rtree=calculate_mape(target,predicted_rtree)\n",
    "    return predicted_rtree\n",
    "\n",
    "def run_abr(predictors,target):\n",
    "    #rtree = DecisionTreeRegressor(max_depth=25)\n",
    "    rtree = ABR()\n",
    "    predicted_rtree=cross_val_predict (rtree, predictors,target, cv=10)\n",
    "    mape_rtree=calculate_mape(target,predicted_rtree)\n",
    "    return predicted_rtree\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "nytData=pd.read_csv(\"C:\\\\Users\\\\venka\\\\Documents\\\\Fall Semester\\\\Predictive Analytics\\\\Project/nyt_data.txt\",sep='\\t')\n",
    "topics=pd.read_csv(\"C:\\\\Users\\\\venka\\\\Documents\\\\Fall Semester\\\\Predictive Analytics\\\\Project/results DocsToTopics.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 132,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "nytDataV1=nytData.drop('headline_main',axis=1)\n",
    "nytDataV1=nytDataV1.drop('word_count',axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 133,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "nytDataV1 = nytDataV1[nytDataV1['facebook_like_count']!= 0 ]\n",
    "nytDataV1 = nytDataV1[nytDataV1['facebook_share_count']!= 0 ]\n",
    "nytDataV1 = nytDataV1[nytDataV1['twitter']!= 0 ]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 134,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "nytDataV1=nytDataV1[nytDataV1['news'].notnull()]\n",
    "nytDataV1['index']=range(1,len(nytDataV1)+1)\n",
    "topics.columns=['index','topics']\n",
    "nytDataV1=nytDataV1.merge(topics)\n",
    "nytDataV1=nytDataV1.drop('index',axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 135,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "nytDataV1=nytDataV1[nytDataV1['snippet'].notnull()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "nytDataV1['date_collected'] = pd.to_datetime(nytDataV1['date_collected'])\n",
    "nytDataV1['pub_date'] = pd.to_datetime(nytDataV1['pub_date'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 137,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "URL                             object\n",
       "date_collected          datetime64[ns]\n",
       "snippet                         object\n",
       "abstract                        object\n",
       "pub_date                datetime64[ns]\n",
       "news_desk                       object\n",
       "type_of_material                object\n",
       "id                              object\n",
       "text                            object\n",
       "facebook_like_count              int64\n",
       "facebook_share_count             int64\n",
       "googleplusone                    int64\n",
       "twitter                          int64\n",
       "pinterest                        int64\n",
       "linkedIn                         int64\n",
       "news                            object\n",
       "topics                           int64\n",
       "dtype: object"
      ]
     },
     "execution_count": 137,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nytDataV1.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 138,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "7.249844934872646"
      ]
     },
     "execution_count": 138,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nytDataV1['days_old'] = nytDataV1['date_collected'] - nytDataV1['pub_date']\n",
    "nytDataV1['days_old']= nytDataV1['days_old'].dt.days\n",
    "nytDataV1['days_old'].mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 139,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "nytDataV1['snippet']=nytDataV1['snippet'].str.lower()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 140,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "#nltk.download('stopwords')\n",
    "#nltk.download('punkt')\n",
    "#nltk.download('wordnet')\n",
    "\n",
    "nytDataT1 = nytDataV1[nytDataV1['topics'] == 1]\n",
    "nytDataT1=nytDataT1[nytDataT1['days_old']>=nytDataT1['days_old'].mean()]\n",
    "nytDataT2 = nytDataV1[nytDataV1['topics'] == 2]\n",
    "nytDataT2=nytDataT2[nytDataT2['days_old']>=nytDataT2['days_old'].mean()]\n",
    "nytDataT3 = nytDataV1[nytDataV1['topics'] == 3]\n",
    "nytDataT3=nytDataT3[nytDataT3['days_old']>=nytDataT3['days_old'].mean()]\n",
    "nytDataT4 = nytDataV1[nytDataV1['topics'] == 4]\n",
    "nytDataT4=nytDataT4[nytDataT4['days_old']>=nytDataT4['days_old'].mean()]\n",
    "nytDataT5 = nytDataV1[nytDataV1['topics'] == 5]\n",
    "nytDataT5=nytDataT5[nytDataT5['days_old']>=nytDataT5['days_old'].mean()]\n",
    "nytDataT6 = nytDataV1[nytDataV1['topics'] == 6]\n",
    "nytDataT6=nytDataT6[nytDataT6['days_old']>=nytDataT6['days_old'].mean()]\n",
    "nytDataT7 = nytDataV1[nytDataV1['topics'] == 7]\n",
    "nytDataT7=nytDataT7[nytDataT7['days_old']>=nytDataT7['days_old'].mean()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 141,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "nytDataT1 = nytDataV1[nytDataV1['topics'] == 1]\n",
    "nytDataT1=nytDataT1[nytDataT1['twitter']>=nytDataT1['twitter'].mean()]\n",
    "nytDataT2 = nytDataV1[nytDataV1['topics'] == 2]\n",
    "nytDataT2=nytDataT2[nytDataT2['twitter']>=nytDataT2['twitter'].mean()]\n",
    "nytDataT3 = nytDataV1[nytDataV1['topics'] == 3]\n",
    "nytDataT3=nytDataT3[nytDataT3['twitter']>=nytDataT3['twitter'].mean()]\n",
    "nytDataT4 = nytDataV1[nytDataV1['topics'] == 4]\n",
    "nytDataT4=nytDataT4[nytDataT4['twitter']>=nytDataT4['twitter'].mean()]\n",
    "nytDataT5 = nytDataV1[nytDataV1['topics'] == 5]\n",
    "nytDataT5=nytDataT5[nytDataT5['twitter']>=nytDataT5['twitter'].mean()]\n",
    "nytDataT6 = nytDataV1[nytDataV1['topics'] == 6]\n",
    "nytDataT6=nytDataT6[nytDataT6['twitter']>=nytDataT6['twitter'].mean()]\n",
    "nytDataT7 = nytDataV1[nytDataV1['topics'] == 7]\n",
    "nytDataT7=nytDataT7[nytDataT7['twitter']>=nytDataT7['twitter'].mean()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 142,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "4.977743668457406"
      ]
     },
     "execution_count": 142,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nytDataT5['days_old'].mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 143,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "#nltk.download('stopwords')\n",
    "#nltk.download('punkt')\n",
    "#nltk.download('wordnet')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 144,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "stop = set(stopwords.words('english'))\n",
    "tokenizer = RegexpTokenizer(r'\\w+')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "to_tfidf1,actual1 = to_tfidf(nytDataT1)\n",
    "to_tfidf2,actual2 = to_tfidf(nytDataT2)\n",
    "to_tfidf3,actual3 = to_tfidf(nytDataT3)\n",
    "to_tfidf4,actual4 = to_tfidf(nytDataT4)\n",
    "to_tfidf5,actual5 = to_tfidf(nytDataT5)\n",
    "to_tfidf6,actual6 = to_tfidf(nytDataT6)\n",
    "to_tfidf7,actual7 = to_tfidf(nytDataT7)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "mape_svr1=run_svr(to_tfidf1,actual1)\n",
    "mape_svr2=run_svr(to_tfidf2,actual2)\n",
    "mape_svr3=run_svr(to_tfidf3,actual3)\n",
    "mape_svr4=run_svr(to_tfidf4,actual4)\n",
    "mape_svr5=run_svr(to_tfidf5,actual5)\n",
    "mape_svr6=run_svr(to_tfidf6,actual6)\n",
    "mape_svr7=run_svr(to_tfidf7,actual7)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 169,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "mape_nn1=run_nn(to_tfidf1,actual1)\n",
    "mape_nn2=run_nn(to_tfidf2,actual2)\n",
    "mape_nn3=run_nn(to_tfidf3,actual3)\n",
    "mape_nn4=run_nn(to_tfidf4,actual4)\n",
    "mape_nn5=run_nn(to_tfidf5,actual5)\n",
    "mape_nn6=run_nn(to_tfidf6,actual6)\n",
    "mape_nn7=run_nn(to_tfidf7,actual7)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "mape_abr1=run_abr(to_tfidf1.toarray(),actual1)\n",
    "mape_abr2=run_abr(to_tfidf2.toarray(),actual2)\n",
    "mape_abr3=run_abr(to_tfidf3.toarray(),actual3)\n",
    "mape_abr4=run_abr(to_tfidf4.toarray(),actual4)\n",
    "mape_abr5=run_abr(to_tfidf5.toarray(),actual5)\n",
    "mape_abr6=run_abr(to_tfidf6.toarray(),actual6)\n",
    "mape_abr7=run_abr(to_tfidf7.toarray(),actual7)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 177,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "mape_rtree1,mape_rtree2,mape_rtree3,mape_rtree4,mape_rtree5,mape_rtree6,mape_rtree7"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 170,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(40.14115057855502,\n",
       " 37.16533236612782,\n",
       " 38.424583227582254,\n",
       " 39.30512349095652,\n",
       " 38.313389103351945,\n",
       " 38.901978228728176,\n",
       " 38.285294386630454)"
      ]
     },
     "execution_count": 170,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mape_svr1,mape_svr2,mape_svr3,mape_svr4,mape_svr5,mape_svr6,mape_svr7"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 178,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Category         MAPE\n",
      "Business      : 40.14%\n",
      "Travel        : 37.17%\n",
      "LifeStyle     : 38.42%\n",
      "Sports        : 39.31%\n",
      "Entertaintment: 38.31%\n",
      "U.S.          : 38.90%\n",
      "Politics      : 38.29%\n"
     ]
    }
   ],
   "source": [
    "print ('Category         MAPE')\n",
    "print ('Business      : {:.2f}%' .format(mape_svr1))\n",
    "print ('Travel        : {:.2f}%'.format(mape_svr2))\n",
    "print ('LifeStyle     : {:.2f}%'.format(mape_svr3))\n",
    "print ('Sports        : {:.2f}%'.format(mape_svr4))\n",
    "print ('Entertaintment: {:.2f}%'.format(mape_svr5))\n",
    "print ('U.S.          : {:.2f}%'.format(mape_svr6))\n",
    "print ('Politics      : {:.2f}%'.format(mape_svr7))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 205,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "stacker=LinearRegression()\n",
    "stacked_result1=pd.DataFrame({'abr':mape_abr1,'svr':mape_svr1,'nn':mape_nn1})\n",
    "predicted_stack=cross_val_predict (stacker, stacked_result1,actual1, cv=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 206,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "73.21175527507158"
      ]
     },
     "execution_count": 206,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mape_stack=calculate_mape(actual1,predicted_stack)\n",
    "mape_stack"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "736.8371236133122"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nytDataV1['twitter'].mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 200,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "abr    0\n",
       "nn     0\n",
       "svr    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 200,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "stacked_result1.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 204,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)"
      ]
     },
     "execution_count": 204,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "stack_fit1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "twitter_mean=pd.DataFrame(nytDataV1[['twitter','topics']].groupby(['topics']).mean())\n",
    "twitter_mean['topic_name']=['Business','Travel','LifeStyle','Sports','Entertaintment','U.S.','Politics']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>twitter</th>\n",
       "      <th>topic_name</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>topics</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>398.995533</td>\n",
       "      <td>Business</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>417.742047</td>\n",
       "      <td>Travel</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>433.130074</td>\n",
       "      <td>LifeStyle</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>418.456368</td>\n",
       "      <td>Sports</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>395.685829</td>\n",
       "      <td>Entertaintment</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           twitter      topic_name\n",
       "topics                            \n",
       "1       398.995533        Business\n",
       "2       417.742047          Travel\n",
       "3       433.130074       LifeStyle\n",
       "4       418.456368          Sports\n",
       "5       395.685829  Entertaintment"
      ]
     },
     "execution_count": 101,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "twitter_mean.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABAAAAAL2CAYAAADfBuS9AAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\nAAAPYQAAD2EBqD+naQAAIABJREFUeJzs3XucVXW9N/DPnhmYATf34TKIKDcVw8wu6lOYkVYEXvDJ\nwkpUNLNM7fb4yqejp4uec+p4Tp3sco75qqeL9nSwzFMioOnj9Rw1U0vyniAieCFwZJBBmNnPH76a\n4wjoMA5shvV+v168XrN/67fW+q714wezP3uttUuVSqUSAAAAYJdWU+0CAAAAgO1PAAAAAAAFIAAA\nAACAAhAAAAAAQAEIAAAAAKAABAAAAABQAAIAAAAAKAABAAAAABSAAAAAAAAKQAAAQGHU1NTk7LPP\nrnYZPeKmm25KTU1NrrzyymqXAgD0EgIAAHq9++67L8cdd1z22muv9OvXL2PGjMl73/vefOc736l2\nadtVqVSqdgnb5IEHHshXvvKVLFu2rNqlAEAhCQAA6NX+8z//M29729ty33335eMf/3i++93v5rTT\nTkttbW0uvvjiape3XVUqlWqXsE3uv//+fOUrX8nSpUurXQoAFFJdtQsAgNfj7/7u7zJ48ODcdddd\nGTBgQKdlq1at2uH1vPDCC+nfv/8O329vUKlUesVVC5VKJS+++GLq6+urXQoA9ChXAADQqz322GN5\nwxvesNmb/yRpbGzc4jr/8R//kf333z8NDQ2ZMmVKFi1a1Gn5smXLcsYZZ2TfffdN//7909jYmA99\n6EN5/PHHO/X78Y9/nJqamtx8880544wzMnLkyOyxxx4dy1esWJFTTjklo0aN6tjX//k//2ezer79\n7W9nypQp2W233TJ06NC87W1vy89//vPXPPZSqZS2trZ88YtfTFNTU8rlco455pgsX768o8+Xv/zl\n9O3bN3/5y182W//jH/94hg4dmhdffPFV9/PQQw/lQx/6UEaMGJH+/ftn3333zXnnnbdN5+vHP/5x\nPvShDyVJ3vWud6Wmpia1tbW5+eabO/osWLAg73znO1MulzNw4MAceeSRuf/++zer54orrsgb3vCG\n9OvXL2984xtz1VVX5eSTT864ceM69XvhhRfy+c9/PmPHjk1DQ0P23Xff/PM///Nm2/vrsyF+9rOf\nZcqUKWloaMiCBQsybty4HHvssZv137BhQwYNGpRPfvKTr3reAGBn4woAAHq1PffcM7fffnv+9Kc/\n5Q1veMNr9r/lllty5ZVX5owzzsiAAQNy8cUX57jjjsuyZcsyZMiQJMnvfve73H777fnwhz+cMWPG\nZOnSpfne976XadOm5f77709DQ0OnbZ5xxhkZMWJEvvSlL2XdunVJkmeeeSYHH3xwamtrc/bZZ6ex\nsTELFizIqaeemrVr13Y8jPDSSy/Npz/96XzoQx/KZz7zmbS2tuaPf/xj7rjjjhx//PGveiyVSiUX\nXnhhampqcu655+aZZ57JN7/5zbznPe/Jvffem/r6+syZMydf/epX8+///u8544wzOtbduHFjfvnL\nX+a4445L3759t7qPP/7xjzn00ENTX1+f008/PXvuuWf+/Oc/5+qrr86FF17Y5fN12GGH5eyzz863\nv/3tnHfeedl3332TJJMnT06S/PSnP83JJ5+c6dOn5x//8R/zwgsv5F//9V9z6KGH5p577snYsWOT\nJPPnz8/xxx+fAw44IF/72teyZs2anHrqqdl99903u7rgqKOOyk033ZSPfexjOeCAA7Jo0aKcc845\nWbFixWZBwPXXX5958+blzDPPTGNjY8aPH58TTjghF110UZ577rkMHjy4o++vf/3rtLS0ZM6cOa86\nPgCw06kAQC923XXXVfr06VOpq6urvP3tb6984QtfqFx77bWVjRs3bta3VCpVGhoaKkuWLOlo++Mf\n/1gplUqV7373ux1tra2tm617xx13VEqlUuWyyy7raPvRj35UKZVKlcMOO6zS3t7eqf+pp55a2X33\n3Str1qzp1P7hD3+4MmTIkI59zJo1q7L//vtv83HfeOONlVKpVNljjz0q69at62i/4oorKqVSqfLt\nb3+7o+3tb3975X/8j//Raf0rr7yyUlNTU7n55ptfdT/vfOc7K4MGDaosX758q326er5+8YtfVGpq\naio33XRTp74tLS2VIUOGVD7xiU90an/mmWcqgwcPrpx++ukdbfvvv39l7NixlRdeeKGj7eabb66U\nSqXKuHHjOtquuuqqSqlUqvzDP/xDp21+8IMfrNTW1lYee+yxjrZSqVSpq6urPPjgg536Pvzww5VS\nqVS55JJLOrUfffTRlfHjx2/1fADAzsotAAD0akcccUT+67/+K8ccc0z++Mc/5qKLLsr73ve+7L77\n7vnNb36zWf/3vOc92WuvvTpe77///hk4cGAee+yxjraX3/u9adOmrF69OuPHj8/gwYNz9913d9pe\nqVTKaaedttmnz1deeWWOOuqotLW15S9/+UvHn/e+97157rnnOrYzePDgLF++PHfddVe3jv+kk07q\n9MyB4447Lk1NTbnmmms62k488cTccccdWbJkSUfb5Zdfnj322COHHnroVre9atWq3HLLLR2fsG/N\ntpyvLbnuuuvS3Nyc448/vtO5KpVKOfjgg/P//t//S5KsXLkyixcvzkknnZR+/fp1rH/ooYdm//33\n77TNBQsWpK6uLmeddVan9s9//vNpb2/PggULOrW/613vyj777NOpbdKkSTn44INz+eWXd7StWbMm\nCxcuzAknnPCaxwUAOxsBAAC93lve8pb84he/yJo1a3LnnXfmi1/8YlpaWvLBD34wDz74YKe+L79H\n/6+GDBmSNWvWdLxubW3N3/7t32bs2LGpr69PY2NjRowYkebm5jQ3N2+2/ssDhSR59tln89xzz+X7\n3/9+hg8f3unPKaecklKplGeeeSZJ8oUvfCHlcjkHHXRQ9t5775x55pn5z//8zy4f+8SJE7fY9vIn\n7c+ePTt9+/bteCP7/PPPZ/78+a/5Jvavochr3VqxrefrlR555JFUKpVMmzat07kaMWJErrvuujz7\n7LNJ0vFMgQkTJmzxmF/u8ccfz+jRo7Pbbrt1av/rLQevfJ7DK8fwr0488cTcdttteeKJJ5Ik8+bN\ny6ZNmwQAAPRKngEAwC6jrq4ub3nLW/KWt7wlkyZNyty5c3PFFVfk/PPP7+hTW1u7xXUrL/tKvTPP\nPDM//vGP89nPfjaHHHJIBg0alFKplNmzZ6e9vX2zdV/+aXSSjj4nnHBCTjrppC3u741vfGOSZN99\n981DDz2Uq6++OgsXLsyVV16Z733ve/nSl76UL33pS9t2ArZi8ODBOfLII3P55ZfnvPPOyxVXXJEX\nX3wxH/3oR3tk+9t6vl6pvb09pVIpl112WUaOHLnZ8rq67f/ryivH8K+OP/74fPazn83ll1+ec889\nN5dffnne+ta3ZtKkSdu9JgDoaQIAAHZJb33rW5O8dNn4tvrlL3+Zk08+Of/4j//Y0bZhw4Y899xz\nXVp/+PDhGTBgQNra2vLud7/7Nfv369cvH/zgB/PBD34wmzZtyrHHHpu/+7u/y//+3//7VR/Ql7z0\n6fkrPfrooznggAM6tZ144omZNWtW7rrrrvzsZz/LgQce2PFp+NaMHz8+SbJ48eJX7dfV87W1rwCc\nMGFCKpVKhg8f/qrna88990zy0vG90ivb9txzz1x//fVZt25dp6sAHnjggU7bei1DhgzJzJkzc/nl\nl+cjH/lIbrvttlx88cVdWhcAdjZuAQCgV7vxxhu32D5//vwk2ey+7q6ora3d7JPriy++OG1tbV1a\nv6amJh/4wAfyy1/+Mn/60582W75q1aqOn1evXt1pWV1dXSZPnpxKpZKNGze+5r5+8pOfpKWlpeP1\nFVdckZUrV2bGjBmd+r3//e/PsGHD8vWvfz033XRTl55g39jYmHe+85354Q9/2HEJ/JZ09Xzttttu\nqVQqmwUD73vf+zJw4MD8/d//fTZt2rTZ9v96vpqamjJlypT85Cc/yQsvvNCx/Kabbsp9993XaZ0Z\nM2Zk06ZN+c53vtOp/Zvf/GZqamry/ve//1WOvLM5c+bkT3/6U84555zU1dVl9uzZXV4XAHYmrgAA\noFc766yz8sILL+TYY4/NvvvumxdffDG33XZb5s2bl/Hjx2fu3LnbvM0jjzwyP/3pTzNw4MDst99+\n+a//+q9cf/31aWxs3Kzvy28deLmvfe1rufHGG3PwwQfntNNOy3777ZfVq1fn97//fW644YaON7Xv\nfe97M2rUqLzjHe/IyJEjc//99+e73/1ujjzyyM3uX9+SoUOHZurUqZk7d26eeuqpfOtb38ree++d\nj33sY5361dXV5fjjj893vvOdjp+74uKLL86hhx6aN7/5zfn4xz+ecePGZcmSJbnmmmtyzz33bNP5\netOb3pTa2tp8/etfz3PPPZf6+vocfvjhaWxszL/+67/mxBNPzJvf/OYcf/zxGT58eJYtW5b58+dn\n6tSpHZ+6//3f/31mzZqVt7/97Zk7d25Wr16d7373u9l///07BSFHHXVUpk2blr/5m7/JkiVLOr4G\n8De/+U0++9nPZty4cV06/iSZOXNmhg0bliuuuCIzZszY4t8DAOgVqvkVBADwei1atKjysY99rLLf\nfvtVBg4cWGloaKjsvffelc985jOVZ599tlPfmpqaytlnn73ZNsaNG1c55ZRTOl43NzdXTj311MqI\nESMqAwcOrMyYMaPy8MMPb9bvRz/6UaWmpqby+9//fou1Pfvss5Wzzjqrsueee1bq6+sro0ePrrzn\nPe+p/OAHP+joc+mll1be9a53VYYPH17p169fZdKkSZVzzz23snbt2lc97htvvLFSU1NT+fd///fK\n3/zN31RGjRpV2W233SpHH3105YknntjiOr/73e8qpVKp8v73v/9Vt/1K999/f+UDH/hAZejQoZX+\n/ftXJk+eXPnyl7/csbyr56tSqVR+8IMfVCZOnFjp06fPZl8JeNNNN1Xe//73V4YMGVLp379/ZdKk\nSZVTTjmlcvfdd3faxrx58yr77bdfpaGhoTJlypTKf/zHf1SOO+64yn777dep37p16yqf//znK2PG\njKnU19dX9tlnn8o3vvGNzY5va38vXu5Tn/pUx/kGgN6qVKls5aMLAGCX8sc//jFvetObctlll+Uj\nH/lItcvpUQceeGBGjBiRRYsWbZftf+5zn8sPf/jDPPXUU2loaNgu+wCA7c0zAACgIL7//e9nwIAB\nOfbYY6tdSrdt2rRps2cL3HjjjfnDH/6QadOmbZd9btiwIZdddlmOO+44b/4B6NU8AwAAdnFXX311\n/vSnP+XSSy/N2WefvdWvvOsNnnzyyRxxxBE54YQTMnr06DzwwAO55JJLMnr06Jx++uk9uq9nn302\n1113XX7xi19k9erVOfvss3t0+wCwo7kFAAB2cePGjcszzzyT6dOn5yc/+UmXHi64s3r++edz+umn\n57bbbsuzzz6b3XbbLUcccUT+4R/+YZse7NcVN910U6ZNm5aRI0fmb//2b/PJT36yR7cPADuaAAAA\nAAAKwDMAAAAAoAAEAAAAAFAAAgAAAAAoAAEAAAAAFIAAAAAAAApAAAAAAAAFIAAAAACAAhAAAAAA\nQAEIAAAAAKAABAAAAABQAAIAAAAAKAABAAAAABSAAAAAAAAKQAAAAAAABSAAAAAAgAIQAAAAAEAB\nCAAAAACgAAQAAAAAUAACAAAAACgAAQAAAAAUgAAAAAAACkAAAAAAAAUgAAAAAIACEAAAAABAAQgA\nAAAAoAAEAAAAAFAAAgAAAAAoAAEAAAAAFIAAAAAAAAqgrtoF7MxuueWWXH/99TnkkEMyffr0JMlV\nV12Ve++9t1O/iRMn5oQTTuh4vWnTpixatCiLFy9OW1tbJkyYkJkzZ6ZcLu/Q+gEAAOCvBABb8eST\nT+b3v/99Ro0atdmySZMmZdasWalUKkmSurrOp3HhwoV55JFHMnv27NTX12f+/PmZN29eTjnllB1S\nOwAAALySWwC2YMOGDbnyyitz9NFHp6GhYbPltbW12W233VIul1Mulzv1aW1tzT333JPp06dnr732\n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\n+c1v5sADD+yxIgEAAIDXp1sBQJLU19fnLW95y2btAwcOzAc+8IHXVRQAAADQs7r1NYDJS5/0/9M/\n/VNmzpyZt73tbfnd736XJFm9enUuvvjiPPbYYz1WJAAAAPD6dOsKgBUrVuSwww7L0qVLM378+Dz6\n6KNZu3ZtkmTo0KH59re/ncceeyz/8i//0qPFAgAAAN3TrQDgnHPOyZo1a3L33XenqakpI0aM6LT8\n2GOPzfz583ukQAAAAOD169YtAAsXLsynP/3p7L///imVSpstHz9+fJ544onXXRwAAADQM7oVAKxf\nv36zT/1frqWlpdsFAQAAAD2vWwHAfvvtl1tvvXWry3/961/nTW96U7eLAgAAAHpWtwKAs846Kz/7\n2c/yz//8zx0P/0uSpUuXZu7cubntttvymc98pseKBAAAAF6fbj0E8KSTTsrSpUvzhS98Ieeee26S\nZPr06Wlra0upVMpXv/rV/M//+T97tFAAAACg+7oVACTJl770pcyZMye//OUv8+ijj6a9vT0TJkzI\nBz7wgUyaNKknawQAAABep20OAFpbW/PDH/4wb3zjGzN16tScc84526MuAAAAoAdt8zMAGhoa8rnP\nfS4PPPDA9qgHAAAA2A669RDAKVOm5PHHH+/pWgAAAIDtpFsBwIUXXph/+7d/y4033tjD5QAAAADb\nQ7ceAvj9738/w4YNy+GHH55JkyZl3Lhx6devX6c+pVIpv/zlL3ukSAAAAOD16VYAcOedd6ZUKmX0\n6NFZt25dFi9evFmfUqn0uosDAAAAeka3AoDly5f3dB0AAADAdtStZwAAAAAAvUu3rgB4ufXr16e5\nuTnt7e2bLRs9evTr3TwAAADQA7odAFx66aX5xje+kYcffnirfdra2rq7eQAAAKAHdesWgEsvvTSn\nn356xowZk6985SupVCo566yz8r/+1//KiBEjcsABB+SSSy7p6VoBAACAbupWAPCtb30r73nPe3Ld\nddflk5/8ZJLk6KOPzte//vXcf//9aW5uTktLS48WCgAAAHRftwKAP//5zznmmGOSJH369EmSvPji\ni0mSIUOG5LTTTst3vvOdHioRAAAAeL269QyAgQMHZtOmTR0/9+/fP0888USn5StWrOiZCnewW265\nJQ888EBWrVqVPn36ZI899sgRRxyRxsbGTv1uuOGG3H333Wltbc3YsWMzc+bMDBs2rGP5pk2bsmjR\n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\n6NGjMplMioiI0MCBAxUYGChJOnTokAYOHKjIyEhJUmBgoMLDw+3vkZiYqNzcXK1Zs0arV6+WyWTS\n73//e3344YcaOnSofR9K16/ELFq0SL/97W9vWe+5c+e0du1aZWRkyNPT036VxtfXtxr3kuvd2Ie7\ndOmiAwcO6Mcff1Tv3r31ww8/KCUlRdnZ2fLz81O3bt3Us2fPStd1Y5t/8sknMplM9ts9mjZtqvHj\nx2vx4sW6du2afdSIzWbT5s2btWvXLl26dMn+GdenT587fv4ZUUFBgTIyMjRhwgQ1adJEkhQQEKCG\nDRval5kyZYri4+P1448/6vjx4/Lz81NcXFy5/nDu3DmtXr1aJ0+elKenp9q2bav+/fvbPxfLRvc0\naNBAO3bskIeHhwICAir0t7fffvu234tGdeN3fNnxxM3flUeOHNGsWbP0zDPPaO3atbpw4YImTJgg\nb29vJSUl6fTp0yoqKlL9+vX16KOPqmnTppKkNWvWKDMzUxMmTCi3vs8++0ydOnVSr169JEk7duxQ\nWlqaLl26pKCgIHXv3l2dO3euxq3GjQ4dOqSIiIhyn5nBwcFq3bq1C6sCXIcAAPeFa9euac+ePapb\nt658fX1lMpluecXqxmkrV65UaWmpnn32WXl6eurChQvlkvybX19UVKStW7fqiSeekCQtWrRISUlJ\n+tnPfiZJ2rt3r1JTUxUfH6+wsDCdOXNGy5cvl9lsVocOHZSWlqZDhw7pqaeeUkBAgC5duqTLly9L\nkvbv36+0tDQ9+eSTqlevnqxWq86dO+f0/XQvNmzYoLi4OD322GPatm2bFi5cqN/85jfy9/fXiBEj\nNG/ePP3617+W2Wy2nxB/8803+v777/X444+rbt26OnHihBYtWqQ6derYD3glaf369XrssccUFBQk\nDw8P9ezZU0eOHNG4ceNks9nsB10mk0mDBg1SYGCgcnJytHLlSq1du1bx8fGV1p2dna0ff/xRo0eP\nVn5+vubNm6dNmzYpNjZWAwYM0MWLF1W/fn3FxsbKZrOpTp06ys3NtW9zfHy8PD09NW/ePM2fP18e\nHh4aPny4CgsL9dVXX2n79u32g7Sqbm9ycrL69+8vX19frVixQkuXLtWzzz6rdu3a6fz587fc9vvF\na6+9ppkzZyoyMlI9e/aU2WxWSUmJZs2apcaNG+vZZ5+Vm5ubNm7cqFmzZumFF16wj6g5fPiwWrdu\nfcsrxyNGjNA//vEPdenSRZ06dZJ0/cpadHS0du/eXe6EZ/fu3WrXrp3MZnOFK6UFBQWaOXOmOnfu\nrIEDB6qoqEhr167V/PnzNW7cuOrdOfcZDw8P5efnKzMzU/Pnz1dMTIzatWunkydPauXKlfL19a1S\nuPTzn/9c06dP17hx41SvXj371eObPyPXrVunXbt2acCAAYqIiNCVK1d04cIFSbrt559Rmc1mmc1m\nHTx4UA0bNqx09FJKSori4uI0cOBA7dmzRwsWLNCkSZMUEhKiwsJCe9+bOHGirFarli1bpq+//loJ\nCQn2daSnp8vLy0tjx46VdP2k9ub+Jt35exG3t379eg0YMED+/v7y9fVVdna2Wrdurbi4OLm5uWnX\nrl2aM2eOfv3rX8tisah9+/ZKS0vT5cuX5e/vL0k6ffq0srOz1b59e0nSrl27tGnTJg0aNEihoaHK\nzMzUsmXL5O3trXbt2rlycw3DYrHo+++/1/nz51W/fn1XlwO4HAEAXObQoUN6//33JUmFhYXy8/PT\nqFGjqvz6y5cvq23btvYP86CgoNsuX1paqsGDB9uX69atmzZs2GCfn5qaqv79+9sT4cDAQF24cEHf\nfvutOnTooMuXLys4OFgRERGSVG749OXLl+Xn56fmzZvb79+98SpQTbhxf5bp06eP+vTpI0nq2LGj\noqOjJUn9+vXTtm3bdPr0aUVGRtrvFfb19bWfsBYXF+ubb77RuHHj1KhRI0nX93FGRoa+/fbbcifE\nMTExat68uf1vs9ksNze3CldZbrxXNTAwULGxsVqxYsVtAwBJSkhIsB/EdujQQenp6YqNjZW3t7fc\n3d3l6elZ4b1MJpNiY2Pt9+F26tRJ69ev18svv2y/qt22bVsdO3ZMvXr1qvL2mkwm9evXz/537969\nNWfOHBUXF8vT07PSbb9fWCwWubm5yWw226+M7d27VzabTUOGDLEvN2TIEP3pT3/S8ePH1aJFCz3+\n+ONatGiR/vznPys0NFQRERFq27atvT/4+PhUWK90fb//61//ktVqlcVi0ZUrV3T48OFKT+a3b9+u\n8PBwxcbGlqvlr3/9qy5evKjg4ODq2C33naNHj+ro0aN66KGHtHXrVjVv3lwPP/ywpOtXri5cuKAt\nW7ZUKQAo+7fo4+NT6a1R165d07Zt2xQfH68OHTpIuv7vv6wv3O7zz6jc3Nw0bNgwLVu2TDt27FB4\neLiaNm2q6Ojocs+yadeunR588EFJUmxsrNLT0+37+vvvv1dxcbGGDRsmT09P1atXT4MGDdKc9LBp\nyQAAFZlJREFUOXMUFxdnbzuz2awhQ4aUG/p/q/52t9+L+K+yz/ayq/uS1KBBAzVo0MD+d1xcnA4c\nOKBDhw6pU6dOatCggYKDg7Vv3z771eV9+/apSZMm9kAgNTVVgwYNsg89DwwM1NmzZ/Xtt98SANSQ\nbt26KSMjQ//4xz8UEBCgRo0aqUWLFmrfvj23HcKQ+FcPl2nWrJkGDx4sm82mgoIC7dixQ7NmzdIv\nfvGLKr3+oYce0ooVK3TkyBE1b95cbdu2ve0DBD09PcsdDJWdjEjXA4js7GwtXbpUy5Ytsy9TWlpq\nPyHu2LGjZs6cqc8++0yRkZFq2bKlWrRoIen6iWRaWpo+/vhjRUZGKioqSq1atarR+wdv3J9lbnzw\n2437xmw2y8vLy779t5Kdna2ioiLNnDmz3PSSkpJyQ79NJlO5A6TbOXr0qDZt2qSsrCxdu3ZNpaWl\nKikpUVFRkX3Uwc0CAwPLXcG6sd3u5MZtrlOnjjw9Pe0n/2XTTp8+Lanq23vzessOvq9cuVJrT4rO\nnj2r7OzsCgFScXGxcnJyJElNmjTRyy+/rFOnTunkyZNKT0/Xl19+qZiYGPuJ6a00bNhQ9erV0+7d\nu9W7d2/t2bNHgYGB5QKkm2s5duxYhVpMJpNycnJ+0gFAWYhXUlIiSWrfvr369u2r//f//l+FoaqN\nGzdWWlqa057fkZWVpZKSEjVr1uyW82/3+Wdkbdq0UVRUlDIyMnTq1CkdPnxYmzdv1pAhQ+zhTFmI\nUqZRo0b2EWJZWVkKCwsr9/nXuHFj2Ww2ZWVl2QOA0NDQKt33f7ffiyjv5u+ygoICpaSk6OjRo7Ja\nrSotLVVxcbEuXbpkX6Z9+/bau3evevbsKZvNpn379ikmJkaSdPXqVV2+fFmLFi0qt97S0lLDPafI\nlcxms0aNGqWcnBwdO3ZMp06d0po1a7Rt2zY999xzlR5/AD9VBABwmZtPyB9//HF98MEH2rlz5y0P\nLMsOist06tRJkZGR9vvaN23apP79+6tbt263fL+bT8ZvPGguLCyUdP1K481X7steFx4erldeeUVH\njhxRenq65s+fr+bNm9uHxL700ktKT0/X0aNH9fXXX2vLli2aMGFCjYUAN+/Pm91q+2/38LWyfTJ6\n9Gj5+fmVm3dzYl6VL8/c3Fz95z//UdeuXdWvXz/5+PgoIyNDy5YtU0lJSaXruNu6K3tt2dPVK1vX\n3WzvzeuVaveD7AoLC9WgQQM98cQTFbbjxpEMbm5uioiIUEREhHr16qWNGzdqw4YN6tWr121PTjp1\n6qQdO3aod+/e2r17t/1qaGW1lD0b4uZabm6Xn5qyEM/NzU1+fn41GiDe6SrY7T7/jM7Dw0PNmze3\nj9JYtmyZUlNTnfrsj6qeoNzt9yLKu3k/r1q1SqdPn1ZcXJz9Frc5c+aUOx5p3769UlNTdeHCBeXl\n5Sk/P99+y1PZ98qwYcMqBDE8YNAxXl5eunbtWoXpBQUFd7zlLigoSEFBQerUqZMefvhhffrpp9q/\nf7/Ln9MD1DQCANx3iouL5evrK5vNpry8vHIPAbyZv7+/unTpoi5dumjdunXauXOnQwc6FotFfn5+\nysnJsd+3dyteXl5q166d2rVrpzZt2mj27NnKz8+Xj4+PPDw81LJlS7Vs2VJdu3bV3/72N507d67C\n1eP7UdnJ240nXPXq1ZOHh4cuXbpU6dXa263v5pO3zMxM2Ww29e/f3z5t375991B15e/liHvZ3uqo\npyaFh4dr//798vX1vasnw4eEhNiviLm7u1e67Q888IDWrVunbdu2KSsryz7EvLJaDhw4oICAAMMd\nIFcW4oWEhCgjI6PctIyMDAUHB1fp6v+t+vfNgoOD5eHhofT09HL3lN/odp9/+K+QkBAdPHjQ/vep\nU6fK/Zs/deqU/XshJCREu3fvLjcKKiMjQ25ubgoJCbnt+1TW35z1vQjp5MmT6tKli334fn5+foVn\nX9StW1eNGjXS3r17ZbVaFRUVZT9uCQgIkK+vr3JyctSmTZsar/+nKDg4+JZP78/MzLyrEWIBAQHy\n9PS0hzSAkRAAwGVKSkpktVolXf9S3b59u4qKitSqVSvVrVtXAQEBSk1NVWxsrC5evKitW7eWe/3q\n1asVGRmp4OBg5efn6/jx46pXr559/t2ehMXExGjVqlXy8vJSZGSkiouLlZmZqYKCAvXo0UNbt26V\nxWKxH7jt379fFotFPj4+2r17t0pLS9WoUSN5enpq7969FYabV7cb92cZNze3Kj05vWzo+o8//qio\nqCh5enrKy8tLPXv21OrVq1VaWqqIiAhdu3ZNGRkZ8vb2th/Q3mo/lz3k7+zZs/L395fZbFbdunVV\n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\nMgIAAABq0ODBgzV9+nRt27bttg8CXL58uQoLC7V8+XI1bNjQPn39+vUVlr35QX1lWrRoIavVqpiY\nmNvW1KRJE6WmpqqgoKDcKIDDhw9XWE6SfvzxxwrrOHjwoEJCQm55wn8rY8eO1UcffaTly5fr66+/\nVv369fXYY49V6bUAAMAxPAMAAIAa9Lvf/U6+vr56/vnn7UPwb5Senq5PP/1U7u7uklTuSv+lS5eU\nmJhY4TV16tRRbm5uhelPPfWUtm7dqqSkpArzLl26ZF93//79VVhYqOnTp9vn22w2ff755+XChbCw\nMHXs2FEzZsywP89Akvbt26ekpCTFx8dXYQ9c1759e7Vv317Tp0/XwoUL9fTTT8vNjcMSAACqEyMA\nAACoQc2bN9ecOXM0cuRItWnTRmPHjlV0dLQKCwu1efNmLViwQM8++6xeeeUVeXp6avDgwZo4caLy\n8vL0xRdfKDQ0VGfPni23zs6dO2vatGl67733FBkZqfr16ysmJka//e1vtWzZMg0ePFjjx49X586d\ndeXKFe3du1eLFi3S8ePHVbduXSUkJKhbt2569dVXdfjwYbVu3VrLli2zhwo3hgBTp07VoEGD1L17\ndz333HO6evWq/va3vykoKEhvv/32Xe2LsWPH6rXXXpPJZNLo0aPvfecCAIDbMtludVMgAACoVkeP\nHtXUqVO1du1aZWZmymw2Kzo6Wk8//bR+8Ytf2O/3nzx5sg4dOqSwsDBNmjRJwcHBeu6553Ts2DFF\nRERIks6fP6/nn39eGzduVF5enh555BElJydLkq5evar3339f8+fPV0ZGhvz9/dWyZUs98cQTeuml\nl+wjDbKzs/Xyyy9r+fLlcnNz09ChQ/X888+rT58+mjt3rp588kl77SkpKXr77be1a9cueXp6qm/f\nvvrggw/UqlUr+zLvvPOO/vd//1cXLlxQ3bp1b7kPzp07p0aNGikqKko//PBDde1qAADw/yMAAAAA\nt7RkyRI98cQT2rRpk3r06OH09V+8eFHh4eGaMmWK3nzzTaevHwAAlMfNdgAAQAUFBeX+Li0t1Wef\nfab/r507NHIQjKIw+hwOH08FcczEUAJlwEzaAEcLEQhUaoiEBnD0kWCYdWvWkjX/OQ1c/828l+d5\nXK/Xr2w+Ho84jsP3fwD4J34AAABxv9/j/X5HWZax73s8n89YliX6vo8sy07der1esa5rdF0XdV3/\nnjIAAN/lBAAAiGmaYhiG2LYtPp9PFEURbdtG0zSnb1VVFfM8x+12i3Ec43K5nL4BAPwlAAAAAEAC\n/AAAAACABAgAAAAAkAABAAAAABIgAAAAAEACBAAAAABIgAAAAAAACRAAAAAAIAECAAAAACRAAAAA\nAIAE/AAzedFNunpDpQAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x1c2583875f8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<ggplot: (120888585662)>"
      ]
     },
     "execution_count": 114,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ggplot(aes(x=\"topic_name\", weight=\"twitter\"), twitter_mean) + geom_bar() + xlab(\"Category\") + ylab(\"Average Shares\") +ggtitle(\"Shares by category\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernelspec": {
   "display_name": "Python [default]",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
