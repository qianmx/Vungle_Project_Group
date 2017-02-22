import pandas as pd
import numpy as np

def combine_meta_data(df,url_meta_iso,url_meta_android,url_application):
    meta_iso =  pd.read_csv(url_meta_iso,sep='|',quotechar='"',header=None,error_bad_lines=False)
    meta_android =  pd.read_csv(url_meta_android,sep='|',quotechar='"',header=None,error_bad_lines=False)
    application = pd.read_csv(url_application,header=None,sep='|',quotechar='"',error_bad_lines=False)
    
    application.columns =['vungle_id','market_id','is_publisher','platform'] 
    meta_iso.columns=['market_id','title','created_at','last_updated_at',\
                      'version','size','developer','developer_website',\
                      'market_url','languages','content_rating','genre_ids',\
                      'genres','current_version_user_ratings',\
                      'n_current_version_user_ratings','user_rating',\
                      'n_user_ratings','price_currency_code','price_value',\
                      'min_os_version','supported_devices','screenshot_urls'
                     ]  
    meta_android.columns=['market_id','package_name','title','created_at',\
                      'last_updated_at','version','size','website',\
                      'developer','market_url','languages','content_rating',\
                      'primary_category','categories','user_rating',\
                      'n_user_ratings','price_currency_code','price_value',\
                      'n_min_downloads','n_max_downloads','has_in_app_purchases',\
                      'screenshot_urls','similar'
                     ]
    
    meta_iso.market_id=meta_iso.market_id.astype('string')
    application.market_id=application.market_id.astype('string')
    a = pd.merge(application,meta_iso,on='market_id',how='left')
    a = pd.merge(a,meta_android,on='market_id',how='left',suffixes=['','_android'])
    common= [i for i in a.columns if '_android' in i]
    common_iso = ['title','created_at','last_updated_at','version',\
                  'size','developer','market_url','languages',\
                  'content_rating','user_rating',\
                  'n_user_ratings','price_currency_code','price_value',\
                  'screenshot_urls']
    for i in range(len(common)):
        a[common_iso[i]].fillna(a[common[i]], inplace=True)
        del a[common[i]]
    
    
    columns = a.columns.values
    columns[0]='advertiser_app_store_id'
    a.columns=columns
    b = pd.merge(df,a,on='advertiser_app_store_id',how='left')
    columns = a.columns.values
    columns[0]='publisher_app_store_id'
    a.columns=columns
    b = pd.merge(b,a,on='publisher_app_store_id',how='left',suffixes=['','_publisher'])
    
    publisher_name = [i for i in b.columns if '_publisher' in i][1:]
    keep_col_name = ['_'.join(i.split('_')[0:-1]) for i in publisher_name] 
    b.loc[b.is_publisher=='t',keep_col_name] = np.nan
    for i in range(len(publisher_name)):
        b[keep_col_name[i]].fillna(b[publisher_name[i]], inplace=True)
        del b[publisher_name[i]]

    return b




    