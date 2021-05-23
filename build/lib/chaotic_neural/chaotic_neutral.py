import feedparser
import urllib
from tqdm import tqdm # progress bars!
import smart_open

import collections
from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
import numpy as np                                  # array handling
import matplotlib.pyplot as plt
from summa import summarizer
import pickle
import gensim
import scipy.io as sio
import plotly.graph_objects as go

# affiliations collected using Andy Casey's ADS api: https://github.com/andycasey/ads
# import ads


#---------------------------------------------------------------------
#-------------------------ArXiv querying------------------------------
#---------------------------------------------------------------------

def run_simple_query(search_query = 'all:sed+fitting', max_results = 10, start = 0):
    """
        Query ArXiv to return search results for a particular query
        Parameters
        ----------
        query: str
            query term. use prefixes ti, au, abs, co, jr, cat, m, id, all as applicable.
        max_results: int, default = 10
            number of results to return. numbers > 1000 generally lead to timeouts
        start: int, default = 0
            start index for results reported. use this if you're interested in running chunks.
        Returns
        -------
        feed: dict
            object containing requested results parsed with feedparser
        Notes
        -----
            add functionality for chunk parsing, as well as storage and retreival
        """

    # Base api query url
    base_url = 'http://export.arxiv.org/api/query?';

    query = 'search_query=%s&start=%i&max_results=%i' % (search_query,
                                                     start,
                                                     max_results)

    # Opensearch metadata such as totalResults, startIndex, 
    # and itemsPerPage live in the opensearch namespase.
    # Some entry metadata lives in the arXiv namespace.
    # This is a hack to expose both of these namespaces in
    # feedparser v4.1
    feedparser._FeedParserMixin.namespaces['http://a9.com/-/spec/opensearch/1.1/'] = 'opensearch'
    feedparser._FeedParserMixin.namespaces['http://arxiv.org/schemas/atom'] = 'arxiv'

    # perform a GET request using the base_url and query
    response = urllib.request.urlopen(base_url+query).read()

    # parse the response using feedparser
    feed = feedparser.parse(response)
    
    return feed

def print_feed_info(feed):
    """
    print out feed information
    """

    print('Feed title: %s' % feed.feed.title)
    print('Feed last updated: %s' % feed.feed.updated)

    # print opensearch metadata
    print('totalResults for this query: %s' % feed.feed.opensearch_totalresults)
    print('itemsPerPage for this query: %s' % feed.feed.opensearch_itemsperpage)
    print('startIndex for this query: %s'   % feed.feed.opensearch_startindex)
    
    return

def print_feed_entries(feed, num_entries = 3):
    """
    Run through each entry, and print out information
    """
    
    ctr = 0
    for entry in feed.entries:
        ctr = ctr + 1
        if ctr < (num_entries+1):
            
            print('e-print metadata')
            print('arxiv-id: %s' % entry.id.split('/abs/')[-1])
            print('Published: %s' % entry.published)
            print('Title:  %s' % entry.title)

            # feedparser v4.1 only grabs the first author
            author_string = entry.author

            try:
                print('Authors:  %s' % ', '.join(author.name for author in entry.authors))
            except AttributeError:
                pass

            # get the links to the abs page and pdf for this e-print
            for link in entry.links:
                if link.rel == 'alternate':
                    print('abs page link: %s' % link.href)
                elif link.title == 'pdf':
                    print('pdf link: %s' % link.href)

            # The journal reference, comments and primary_category sections live under 
            # the arxiv namespace
            try:
                journal_ref = entry.arxiv_journal_ref
            except AttributeError:
                journal_ref = 'No journal ref found'
            print('Journal reference: %s' % journal_ref)

            try:
                comment = entry.arxiv_comment
            except AttributeError:
                comment = 'No comment found'
            print('Comments: %s' % comment)

            # Since the <arxiv:primary_category> element has no data, only
            # attributes, feedparser does not store anything inside
            # entry.arxiv_primary_category
            # This is a dirty hack to get the primary_category, just take the
            # first element in entry.tags.  If anyone knows a better way to do
            # this, please email the list!
            print('Primary Category: %s' % entry.tags[0]['term'])

            # Lets get all the categories
            all_categories = [t['term'] for t in entry.tags]
            print('All Categories: %s' % (', ').join(all_categories))

            # The abstract is in the <summary> element
            print('Abstract: %s' %  entry.summary)

            print('\n\n --------------\n\n')

    return

def make_feeds(arxiv_query = 'cat:astro-ph.GA', chunksize = 10, max_setsize = 50000):
    """
    Generate feeds by iteratively querying ArXiv
    """
    
    # first estimate the total number of results
    sq = arxiv_query
    feed = run_simple_query(search_query = sq, max_results=chunksize, start = 0)
    total_results = feed.feed.opensearch_totalresults

    # now iterate in chunks of 100 till we either get the first 5k papers or all the results
    iterate_results = np.amin([int(total_results), max_setsize])
    max_results = chunksize
    num_chunks = int(iterate_results/max_results)

    feeds = []

    for i in tqdm(range(num_chunks)):
        feed = run_simple_query(search_query = sq, max_results=max_results, start = i*max_results)
        feeds.append(feed)
        
    return feeds


#---------------------------------------------------------------------
#--------------------- Building a training set------------------------
#---------------------------------------------------------------------

def read_corpus(feeds, tokens_only=False, titles = False, authors = False):
    ctr = -1
    for nc in tqdm(range(num_chunks)):
        # why do some of the chunks have 0 entries?
        # print(len(feeds[nc].entries))
        
        for i in range(len(feeds[nc].entries)):
            text = feeds[nc].entries[i].summary
            text = text.replace('\n', ' ')
            tokens = gensim.utils.simple_preprocess(text)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags                    
                ctr = ctr + 1
                yield gensim.models.doc2vec.TaggedDocument(tokens, [ctr])
                


def save_trained_doc2vec_model(fname_tag):

    model.save('chaotic_neutral/data/model_'+fname_tag+'_trained.arxivmodel')

    with open("chaotic_neutral/data/titles"+fname_tag+".pkl", "wb") as fp:   #Pickling
        pickle.dump(all_titles, fp)
    with open("chaotic_neutral/data/abstracts"+fname_tag+".pkl", "wb") as fp:   #Pickling
        pickle.dump(all_abstracts, fp)
    with open("chaotic_neutral/data/authors"+fname_tag+".pkl", "wb") as fp:   #Pickling
        pickle.dump(all_authors, fp)
    with open("chaotic_neutral/data/ids"+fname_tag+".pkl", "wb") as fp:   #Pickling
        pickle.dump(all_ids, fp)

    with open('chaotic_neutral/data/train_corpus'+fname_tag+'.pkl', 'wb') as fp:
        pickle.dump(train_corpus, fp)

    with open('chaotic_neutral/data/test_corpus'+fname_tag+'.pkl', 'wb') as fp:
        pickle.dump(test_corpus, fp)
        
    return

def load_trained_doc2vec_model(fname_tag):
    

    model = gensim.models.Word2Vec.load('chaotic_neutral/data/model_'+fname_tag+'_trained.arxivmodel')

    with open("chaotic_neutral/data/titles"+fname_tag+".pkl", "rb") as fp:
        all_titles = pickle.load(fp)
    with open("chaotic_neutral/data/abstracts"+fname_tag+".pkl", "rb") as fp:
        all_abstracts = pickle.load(fp)
    with open("chaotic_neutral/data/authors"+fname_tag+".pkl", "rb") as fp:
        all_authors = pickle.load(fp)

    with open("chaotic_neutral/data/train_corpus"+fname_tag+".pkl", "rb") as fp:
        train_corpus = pickle.load(fp)

    with open("chaotic_neutral/data/test_corpus"+fname_tag+".pkl", "rb") as fp:
        test_corpus = pickle.load(fp)

    return [model, all_titles, all_abstracts, all_authors, train_corpus, test_corpus]


def load_trained_doc2vec_model_mapper(fname_tag):
    
    model = gensim.models.Word2Vec.load('chaotic_neutral/data/model_'+fname_tag+'_trained.arxivmodel')

    with open("chaotic_neutral/data/titles"+fname_tag+".pkl", "rb") as fp:
        all_titles = pickle.load(fp)
    with open("chaotic_neutral/data/abstracts"+fname_tag+".pkl", "rb") as fp:
        all_abstracts = pickle.load(fp)
    with open("chaotic_neutral/data/authors"+fname_tag+".pkl", "rb") as fp:
        all_authors = pickle.load(fp)
    with open("chaotic_neutral/data/ids"+fname_tag+".pkl", "rb") as fp:
        all_ids = pickle.load(fp)

    with open("chaotic_neutral/data/train_corpus"+fname_tag+".pkl", "rb") as fp:
        train_corpus = pickle.load(fp)

    with open("chaotic_neutral/data/test_corpus"+fname_tag+".pkl", "rb") as fp:
        test_corpus = pickle.load(fp)
        
    with open("chaotic_neutral/data/recent_affils.pkl", "rb") as fp:
        recent_affils = pickle.load(fp)
        
    with open("chaotic_neutral/data/recent_latlon.pkl", "rb") as fp:
        [place_names, place_locs] = pickle.load(fp)

    return [model, all_titles, all_abstracts, all_authors, all_ids, train_corpus, test_corpus, recent_affils, place_names, place_locs]


def build_and_train_model(feeds, fname_tag = 'trained_model'):

    train_corpus = list(read_corpus(feeds))
    test_corpus = list(read_corpus(feeds, tokens_only = True))

    all_titles = []
    all_authors = []
    all_abstracts = []
    all_ids = []
    for nc in tqdm(range(num_chunks)):
        for i in range(len(feeds[nc].entries)): 
            all_titles.append(feeds[nc].entries[i].title)
            all_authors.append(feeds[nc].entries[i].authors)
            all_abstracts.append(feeds[nc].entries[i].summary)
            all_ids.append(feeds[nc].entries[i].id)

    model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40,)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    
    save_trained_doc2vec_model(fname_tag)
    
    return model, train_corpus, test_corpus


#---------------------------------------------------------------------
#--------------------Query using trained model------------------------
#---------------------------------------------------------------------

def find_papers_by_author(auth_name):

    doc_ids = []
    for doc_id in range(len(all_authors)):
        for auth_id in range(len(all_authors[doc_id])):
            if auth_name in all_authors[doc_id][auth_id]['name']:
                print('Doc ID: ',doc_id, ' | ', all_titles[doc_id],' | Author entry: ', all_authors[doc_id][auth_id]['name'])
                doc_ids.append(doc_id)

    return doc_ids

def list_similar_papers(model_data, doc_id = [], input_type = 'doc_id', show_authors = False, show_summary = False, return_n = 10):

    model, all_titles, all_abstracts, all_authors, train_corpus, test_corpus = model_data

    if input_type == 'doc_id':
        print('Doc ID: ',doc_id,', title: ',all_titles[doc_id])
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        start_range = 1
    elif input_type == 'arxiv_id':
        print('ArXiv id: ',doc_id)
        arxiv_query_feed = run_simple_query(search_query='id:'+str(doc_id))
        if len(arxiv_query_feed.entries) == 0:
            print('error: arxiv id not found.')
            return
        else:
            print('Title: '+arxiv_query_feed.entries[0].title)
        arxiv_query_tokens = gensim.utils.simple_preprocess(arxiv_query_feed.entries[0].summary)
        inferred_vector = model.infer_vector(arxiv_query_tokens)
        start_range = 0
    elif input_type == 'keywords':
        print('Keyword(s): ',[doc_id[i] for i in range(len(doc_id))])
        word_vector = model.wv[doc_id[0]]
        if len(doc_id) > 1:
           print('multi-keyword')
           for i in range(1,len(doc_id)):
               word_vector = word_vector + model.wv[doc_id[i]]
#         word_vector = model.infer_vector(doc_id)
        inferred_vector = word_vector
        start_range = 0
    else:
        print('unrecognized input type.')
        return
    
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    print('-----------------------------')
    print('Most similar/relevant papers: ')
    print('-----------------------------')
    for i in range(start_range,start_range+return_n):
        
        print(i, all_titles[sims[i][0]], ' (Corrcoef: %.2f' %sims[i][1] ,')')
        if show_authors == True:
            print('Authors:------')
            print(all_authors[sims[i][0]])
        if show_summary == True:
            print('Summary:------')
            text = all_abstracts[sims[i][0]]
            text = text.replace('\n', ' ')
            print(summarizer.summarize(text))
        if show_authors == True or show_summary == True:
            print(' ')
        
    return sims


#---------------------------------------------------------------------
#-------------Geolocation query using trained model-------------------
#---------------------------------------------------------------------

def list_similar_locations(model_data, doc_id = [], input_type = 'doc_id', show_authors = False, show_summary = False, return_n = 10, no_output = False, plt_radius = 10):

    model, all_titles, all_abstracts, all_authors, all_ids, train_corpus, test_corpus, recent_affils, place_names, place_locs = model_data

    if no_output == True:
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        
        
    if input_type == 'doc_id':
        print('Doc ID: ',doc_id,', title: ',all_titles[doc_id])
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        start_range = 1
    elif input_type == 'arxiv_id':
        print('ArXiv id: ',doc_id)
        arxiv_query_feed = run_simple_query(search_query='id:'+str(doc_id))
        if len(arxiv_query_feed.entries) == 0:
            print('error: arxiv id not found.')
            return
        else:
            print('Title: '+arxiv_query_feed.entries[0].title)
        arxiv_query_tokens = gensim.utils.simple_preprocess(arxiv_query_feed.entries[0].summary)
        inferred_vector = model.infer_vector(arxiv_query_tokens)
        start_range = 0
    elif input_type == 'keywords':
        print('Keyword(s): ',[doc_id[i] for i in range(len(doc_id))])
        word_vector = model.wv[doc_id[0]]
        if len(doc_id) > 1:
           print('multi-keyword')
           for i in range(1,len(doc_id)):
               word_vector = word_vector + model.wv[doc_id[i]]
#         word_vector = model.infer_vector(doc_id)
        inferred_vector = word_vector
        start_range = 0
    else:
        print('unrecognized input type.')
        return
    
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    
    year = np.zeros((len(all_ids),))
    month = np.zeros((len(all_ids),))
    num_authors = np.zeros((len(all_ids),))
    for i in range(len(all_ids)):
    # for i in range(100):
        if all_ids[i][21:23] != 'he':
            year[i] = int(all_ids[i][21:23])+2000
            month[i] = int(all_ids[i][23:25])/12
        else:
            all_ids[i]

        num_authors[i] = len(all_authors[i])

    simnums = np.array([sims[i][0] for i in range(len(sims)) if year[sims[i][0]] == 2019 ])

    lats_list = []
    longs_list = []
    # locs_list = []

    good_ids = np.arange(len(all_ids))[year == 2019]

    # print(all_titles[int(simnums[0])])
    # print(int(simnums[0]), simnums[])

    print('----')
    for i in range(return_n):

        arg = np.where(simnums[i] == good_ids)
        affils = recent_affils[int(arg[0])]
        for tempi in range(len(affils)):

            locname = affils[tempi]        
            if (locname != '-'):
                commalocs = [pos for pos, char in enumerate(locname) if char == ',']
                semicolonlocs = [pos for pos, char in enumerate(locname) if char == ';']
                if len(semicolonlocs) > 0:
                    locname = locname[0:semicolonlocs[0]]

                if locname in place_names:                
                    for k in range(len(place_names)):
                        if place_names[k] == locname:
                            lats_list.append(place_locs[k]['lat'])
                            longs_list.append(place_locs[k]['lng'])
    #                 print(locname)
    #                 print(place_locs[locname == place_names])

    #     print(simnums[i], arg, good_ids[arg])
    #     print(all_titles[int(good_ids[arg])])
    #     print(recent_affils[int(arg[0])])

    fig = go.Figure(go.Densitymapbox(lat=lats_list,lon=longs_list, radius=plt_radius))
    fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=0)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()
        
    return

    
    
