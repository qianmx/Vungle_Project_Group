# Machine Learning-Vungle Project Group

### Team Members: 
* Tianyi Liu, Mengxin Qian, and Yixin Zhang.

### Introduciton: 

For online advertising, both publishers and advertisers want to target the users with the most ac- curate advertisements, which would generate high conversion. In our case, we want to help Vungle, the leading in-app video advertising platform for performance marketers, to build a model that can accurately predict whether the impression would convert into a installation.

### Data: 
* **transaction:** This is the main dataset we are using. It has 27columns and 700,000 rows, each represents a specific transaction. The data includes timestamps, ids, and device information.
* **Creatives and Creative tags and Video:** These three tables contains data about the creatives and video attributes, such as created time of the ad and time to show countdown.
* **Ios app metadata and Android apps metadata:** Both tables have metadata about the content associated with vungle id. The content information is either about advertiser app content or the publisher app content. It has 23 columns including user ratings, support languages, etc.
