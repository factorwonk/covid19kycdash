import requests
import pandas as pd
import numpy as np
import string
import warnings

from urllib.request import urlopen
from bs4 import BeautifulSoup as soup
from tqdm import tqdm
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from pylab import bone, pcolor, colorbar, plot, show, rcParams, savefig

class coronavirus_news_aggregator():
    def __init__(self, client_list=[]):
        # Initialize Empty List
        self.client_list = client_list

    def covid19_news_scraper(self, search_query):
        """
        Pass in a client name or search query and returns last 100 headlines associating the client with Covid-19   
        """
        # Use this URL for Australian centric data
        news_url = "https://news.google.com.au/rss/search?q={"+str(search_query)+"%coronavirus}"
        Client = urlopen(news_url)
        xml_page = Client.read()
        Client.close()
        # Beautiful Soup Library is the bomb
        soup_page = soup(xml_page,"xml")
        news_list = soup_page.findAll("item")
        
        # Two separate lists for News Title and Publication Date
        l1 = []
        l2 = []
        for news in news_list:
            # Append to a list
            l1.append(news.title.text)
            l2.append(news.pubDate.text)
            # Zip the two together
            l_tup = list(zip(l1, l2))
        
        # Save this to a DataFrame
        df = pd.DataFrame(l_tup, columns=['Title', 'Date'])
        # Select Date of Headline
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        # Split the Title into Headline and Source columns and then drop the 'Title' column
        df[['Headline','Source']] = df['Title'].str.rsplit("-",1,expand=True)
        df.drop('Title', axis=1, inplace=True)
        df['Client'] = str(search_query)
        return df

    def sentiment_analyser(self, search_query):
        """
        Runs a Google News Search on the input string and then uses VADER sentiment analysis engine on each returned headline.
        Input: Search Query String
        Output: DataFrame with compound sentiment score for each news article
        """
        # Create a Covid-19 News DataFrame for each organization of interest
        news_df = self.covid19_news_scraper(search_query)
        # Initialize VADER Sentiment Intensity Analyzer 
        sia = SIA()
        results = []

        # Calculate the polarity score for each headline associated with the organization
        for row in news_df['Headline']:
            pol_score = sia.polarity_scores(row)
            pol_score['Headline'] = row
            results.append(pol_score)
        
        # Create the Sentiment DataFrame
        sent_df = pd.DataFrame.from_records(results)
        # Merge the two dataframes together on the 'Headline' column
        merge_df = news_df.merge(sent_df, on='Headline')
        # Re-order and Rename the columns
        merge_df = merge_df.rename(columns={'compound':'VADER Score'})
        col_order = ['Client','Date','Headline','Source','VADER Score','neg','neu','pos']
        print('Completed processing %s' % search_query, "...")
        return merge_df[col_order]

    def client_c19_news_agg(self, client_list):
        """
        Provided a list of clients, this pulls up the past 100 covid-19 related news articles on each of them and calculates 
        a Composite Sentiment score for each article related to a client 
        """
        frames = [self.sentiment_analyser(c) for c in client_list]
        result = pd.concat(frames)
        # print()
        # print("VADER Score is a Normalized Weighted Sentiment Composite Score that ranges from +1 (Extremely Positive) to -1 (Extremely Negative)")
        return result

if __name__ == "__main__":
    client_list = ['NAB','CBA','ANZ','Westpac']
    cn = coronavirus_news_aggregator()
    df = cn.client_c19_news_agg(client_list=client_list)
    df.to_csv('client_sentiment_2.csv')
    print("Done!!!")
