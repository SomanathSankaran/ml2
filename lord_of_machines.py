
# coding: utf-8




import pandas as pd
import nltk


# In[43]:


traindf=pd.read_csv("/home/somanath/lordofmachines/train_HFxi8kT/train.csv",encoding='utf8')


# In[44]:


traindf.head()


# In[45]:


traindf.campaign_id.unique().shape


# In[46]:


campaigndf=pd.read_csv("/home/somanath/lordofmachines/train_HFxi8kT/campaign_data.csv",encoding='utf8')


# In[52]:


test=campaigndf[:100]


# In[54]:


test.to_csv("testrec.csv",encoding="utf8")


# In[55]:


traindftot=pd.merge(traindf,campaigndf,how='inner',on="campaign_id")


# In[11]:


traindftot["communication_type"].unique()


# In[5]:


#creating dummy variables
def create_dummies(df,column_name):
    """Create Dummy Columns (One Hot Encoding) from a single Column

    """
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df


# In[76]:


def purify_email_subject(df):
    print(df)
    if  df.lower().rfind('hack'):
        return 1
    else:
        return 0
        


# In[6]:


def purify_email_url(df):
    if  "email_url".lower().rfind('analyticsvidhya'):
        df["email_url_ind"]=1
    else:
        df["email_url_ind"]=0
        


# In[62]:


purify_email_subject(traindftot)


# In[80]:


traindftot[["subject"]]


# In[82]:


traindftot["email_subject_ind"]=traindftot["subject"].apply(purify_email_subject)


# In[63]:


traindftot.subject_ind.unique()


# In[18]:


import re
def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]+',' ', text)


# In[8]:


traindftot=create_dummies(traindftot,"communication_type")


# In[26]:


traindftot.shape


# In[9]:


#making necessary imports for machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV


# In[10]:



def select_features(df):
    # Remove non-numeric columns, columns that have null values
    #df = df.select_dtypes([np.number]).dropna(axis=1)
    all_X = df.drop(["email_body","email_url","subject","is_click","communication_type","send_date","id","is_open"],axis=1)
    all_y = df["is_click"]
    
    clf = RandomForestClassifier(random_state=1)
    selector = RFECV(clf,cv=3)
    selector.fit(all_X,all_y)
    
    best_columns = list(all_X.columns[selector.support_])
    print("Best Columns \n"+"-"*12+"\n{}\n".format(best_columns))
    
    return best_columns


# In[28]:


traindftot=traindftot.drop_duplicates()


# In[48]:


traindftot=traindftot.drop(["send_date"],axis=1)


# In[38]:


traindftot.shape


# In[11]:


cols=select_features(traindftot[:100000])


# In[47]:


traindftot.head(1).T


# In[41]:


def select_model(df,features):
    
    all_X = df[features]
    all_y = df["is_click"]

    # List of dictionaries, each containing a model name,
    # it's estimator and a dict of hyperparameters
    models = [
        #{
        #    "name": "LogisticRegression",
        #    "estimator": LogisticRegression(),
        #    "hyperparameters":
        #        {
        #            "solver": ["newton-cg", "lbfgs", "liblinear"]
        #        }
        #},
    #{
    #    "name": "KNeighborsClassifier",
    #    "estimator": KNeighborsClassifier(),
    #    "hyperparameters":
    #        {
    #            "n_neighbors": range(1,20,2),
    #            "weights": ["distance", "uniform"],
    #            "algorithm": ["ball_tree", "kd_tree", "brute"],
    #            "p": [1,2]
    #        }
    #},
    {
        "name": "RandomForestClassifier",
        "estimator": RandomForestClassifier(random_state=1),
        "hyperparameters":
            {
                "n_estimators": [4, 6, 9],
                "criterion": ["entropy", "gini"],
                "max_depth": [2, 5, 10],
                "max_features": ["log2", "sqrt"],
                "min_samples_leaf": [1, 5, 8],
                "min_samples_split": [2, 3, 5]

            }
     }
    ]
    for model in models:
        print(model['name'])
        print('-'*len(model['name']))

        grid = GridSearchCV(model["estimator"],
                            param_grid=model["hyperparameters"],
                            cv=5)
        grid.fit(all_X,all_y)
        model["best_params"] = grid.best_params_
        model["best_score"] = grid.best_score_
        model["best_model"] = grid.best_estimator_

        print("Best Score: {}".format(model["best_score"]))
        print("Best Parameters: {}\n".format(model["best_params"]))

    return models


# In[42]:


result=select_model(traindftot[:100000],features=cols)


# In[26]:


result


# In[23]:


testdf=pd.read_csv("/home/somanath/lordofmachines/test_BDIfz5B.csv/test_BDIfz5B.csv")


# In[24]:


holdout=pd.merge(testdf,campaigndf,how='inner',on="campaign_id")


# In[39]:


def save_submission_file(model,cols,filename="submission.csv"):
    holdout_data = holdout[cols]
    predictions = model.predict(holdout_data)
    
    holdout_ids = holdout["id"]
    submission_df = {"id": holdout_ids,
                 "is_click": predictions}
    submission = pd.DataFrame(submission_df)

    submission.to_csv(filename,index=False)




# In[28]:


best_rf_model = result[0]["best_model"]


# In[29]:


best_rf_model


# In[40]:


save_submission_file(best_rf_model,cols)


# In[36]:


holdout['communication_type_Webinar']=0


# In[38]:


holdout.columns


# In[65]:


a=str(test["subject"]).lower()


# In[68]:


a

