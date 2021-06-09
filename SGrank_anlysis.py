import json
import string
import requests
from requests.auth import HTTPBasicAuth
import logging
from os import getenv
import textacy
import nltk
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from random import shuffle
import pandas as pd
wordnet_lemmatizer = WordNetLemmatizer()
ARTICLE_TYPE = "article"
CATEGORIES_TYPE = "category"
SLEEP_TIME = 2.0
CSV_HEADERS = ["DOCUMENT", "UNIQUE_ID"]
OVERVIEW_IDENTIFIERS = ["overview", "introduction", "problem"]
ARTICLE_TAGS = ["p", "li"]
OVERVIEW_TAGS = ["h1", "h2", "h3", "h4", "h5", "p"]
EXTRA_CHARS_TO_REMOVE = "\n\r\xa0"
logger = logging.getLogger(__name__)

stop_words = ["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "A", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "after", "afterwards", "ag", "again", "against", "ah", "ain", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appreciate", "approximately", "ar", "are", "aren", "arent", "arise", "around", "as", "aside", "ask", "asking", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "B", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "been", "before", "beforehand", "beginnings", "behind", "below", "beside", "besides", "best", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "C", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "ci", "cit", "cj", "cl", "clearly", "cm", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "could", "couldn", "couldnt", "course", "cp", "cq", "cr", "cry", "cs", "ct", "cu", "cv", "cx", "cy", "cz", "d", "D", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "dj", "dk", "dl", "do", "does", "doesn", "doing", "don", "done", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "E", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "F", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "G", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "H", "h2", "h3", "had", "hadn", "happens", "hardly", "has", "hasn", "hasnt", "have", "haven", "having", "he", "hed", "hello", "help", "hence", "here", "hereafter", "hereby", "herein", "heres", "hereupon", "hes", "hh", "hi", "hid", "hither", "hj", "ho", "hopefully", "how", "howbeit", "however", "hr", "hs", "http", "hu", "hundred", "hy", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "im", "immediately", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "inward", "io", "ip", "iq", "ir", "is", "isn", "it", "itd", "its", "iv", "ix", "iy", "iz", "j", "J", "jj", "jr", "js", "jt", "ju", "just", "k", "K", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "ko", "l", "L", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "M", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu",
              "much", "mug", "must", "mustn", "my", "n", "N", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "neither", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "O", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "otherwise", "ou", "ought", "our", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "P", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "pp", "pq", "pr", "predominantly", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "Q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "R", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "S", "s2", "sa", "said", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "seem", "seemed", "seeming", "seems", "seen", "sent", "seven", "several", "sf", "shall", "shan", "shed", "shes", "show", "showed", "shown", "showns", "shows", "si", "side", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somehow", "somethan", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "sz", "t", "T", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "thereof", "therere", "theres", "thereto", "thereupon", "these", "they", "theyd", "theyre", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "U", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "used", "useful", "usefully", "usefulness", "using", "usually", "ut", "v", "V", "va", "various", "vd", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "W", "wa", "was", "wasn", "wasnt", "way", "we", "wed", "welcome", "well", "well-b", "went", "were", "weren", "werent", "what", "whatever", "whats", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "whom", "whomever", "whos", "whose", "why", "wi", "widely", "with", "within", "without", "wo", "won", "wonder", "wont", "would", "wouldn", "wouldnt", "www", "x", "X", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "Y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "your", "youre", "yours", "yr", "ys", "yt", "z", "Z", "zero", "zi", "zz"]


class ZendeskScraper:

    ARTICLES_ENDPOINT_SUFFIX = "/api/v2/help_center/articles.json"
    CATEGORIES_ENDPOINT_SUFFIX = "/api/v2/help_center/categories.json"
    ARTICLE_WITH_CATEGORY_ENDPOINT_SUFFIX = "/api/v2/help_center/categories/"
    ARTICLE_TYPE = "article"
    CATEGORIES_TYPE = "category"
    SLEEP_TIME = 2.0
    CSV_HEADERS = ["DOCUMENT", "UNIQUE_ID"]
    OVERVIEW_IDENTIFIERS = ["overview", "introduction", "problem"]
    ARTICLE_TAGS = ["p", "li"]
    OVERVIEW_TAGS = ["h1", "h2", "h3", "h4", "h5", "p"]
    EXTRA_CHARS_TO_REMOVE = "\n\r\xa0"

    @classmethod
    def get_response(cls, url, article_type):
        response = requests.get(url, auth=HTTPBasicAuth(
            getenv('USER'), getenv("PASSWORD")))
        if article_type == 0:
            return json.loads(response.text)['article']['body']
        if article_type == 1:
            return response.text
        if article_type == 2:
            return response.text

    @classmethod
    def parse_overview(cls, body):
        soup = BeautifulSoup(body, features="html.parser")
        is_overview_found = False
        overview = []
        for tag in soup.find_all(cls.OVERVIEW_TAGS):
            if not tag.name:
                continue
            elif tag.name == "p" and is_overview_found:
                overview.append(tag)
            elif not is_overview_found and any(
                [identifier in tag.getText().strip().lower()
                 for identifier in cls.OVERVIEW_IDENTIFIERS]
            ):
                is_overview_found = True
            elif not is_overview_found:
                continue
            else:
                break
        return " ".join([para.getText() for para in overview])


def getkeywords(articleUrl, conceptGraphId):

    query = '''query getRelevantConcepts($conceptGraphId: String!, $articleUrl: String!) {
    getRelevantConcepts(conceptGraphId: $conceptGraphId, articleUrl: $articleUrl)
    }
    '''
    variables = {
        "conceptGraphId": conceptGraphId,
        "articleUrl": articleUrl
    }

    url = "https://cncgi-cn-concept-graph-staging.devhub.k8.devfactory.com/graphql/"
    respone = requests.post(url, json={'query': query, 'variables': variables})
    concept_list = json.loads(respone.text)["data"]["getRelevantConcepts"]
    if not concept_list:
        return

    query = '''query getConceptsKeywordsAndArticles($conceptIds: [String!]!) {
    getConceptsKeywordsAndArticles(conceptIds: $conceptIds) {
        name
        keywords {
        processedValue
        }
    }
    }'''
    variables = {
        "conceptIds": concept_list
    }
    respone = requests.post(url, json={'query': query, 'variables': variables})
    result = json.loads(respone.text)["data"]["getConceptsKeywordsAndArticles"]
    article_keywords = []
    for concept in result:
        for word in concept["keywords"]:
            article_keywords.append(word["processedValue"])
    return article_keywords


def genrate_concept(graph_id, exp_id):
    query = '''query getProduct($id: ID){getProduct(id: $id){
    id
    name
    experiments {
      id
      executions(order: {desc: created_at}, first: 1) {
        userLambda
        conceptGraphs {
          id
          concepts{
            id
          }
        }
      }
    }
  }
  }'''
    url = "https://cncgi-cn-concept-graph-staging.devhub.k8.devfactory.com/graphql/"
    vaiables = {'id': graph_id}
    r = requests.post(url, json={'query': query, 'variables': vaiables})
    curr_id = ""
    k = 0
    while curr_id != exp_id:
        curr_id = json.loads(
            r.text)["data"]['getProduct']['experiments'][k]['id']
        k += 1
    concept = json.loads(r.text)["data"]['getProduct']['experiments'][k -
                                                                      1]['executions'][0]['conceptGraphs'][0]['concepts']
    concept_list = list(map(lambda x: x['id'], concept))
    return concept_list


def genrate_url(graph_id, exp_id):
    query = '''query getConceptsKeywordsAndArticles($conceptIds: [String!]!) {
    getConceptsKeywordsAndArticles(conceptIds: $conceptIds) {
        id
        name
        articles {
        id
        title
            url
        }
    }
    }
    '''
    url = "https://cncgi-cn-concept-graph-staging.devhub.k8.devfactory.com/graphql/"
    vaiables = {'conceptIds': genrate_concept(graph_id, exp_id)}
    r = requests.post(url, json={'query': query, 'variables': vaiables})
    data = json.loads(r.text)['data']['getConceptsKeywordsAndArticles']
    articles = []
    ids = {}
    for article_list in data:
        for article in article_list['articles']:
            if article['id'] not in ids:
                articles.append(article)
                ids[article['id']] = 1
    return articles


file1 = open('empty-overview.txt', 'a')


def build_url(url, article_type):
    identifier = list(list(url.split('/'))[-1].split('-'))[0]
    if article_type == 0:
        s = f"https://aureajive.zendesk.com/api/v2/help_center/en-us/articles/{identifier}.json"
    elif article_type == 1:
        s = f"https://support.keriocontrol.gfi.com/hc/en-us/articles/{identifier}.json"
    elif article_type == 2:
        s = f"https://support.archiver.gfi.com/hc/en-us/articles/{identifier}.json"
    return s


en = textacy.load_spacy_lang("en_core_web_sm")


translator = str.maketrans(
    string.punctuation + EXTRA_CHARS_TO_REMOVE,
    " " * (len(string.punctuation) + len(EXTRA_CHARS_TO_REMOVE)),
)
graphids = ["0xdade1", "0x805c1a6", "0x8025508"]
product_id = ["0xd785f", "0x8039344", "0x7cfe37"]
exp_ids = ["0xd9158", "0x803e22b", "0x7f8513"]


def evaluate_article(curr_article, article_type, data):

    print(curr_article)
    url = curr_article['url']
    title = curr_article['title']
    overview = ZendeskScraper.parse_overview(
        ZendeskScraper.get_response(build_url(url, article_type), article_type))

    if not overview:
        print(url)
        file1.write(url)
        return
    overview = (title+". "+overview).lower()
    overview = overview.translate(translator).encode("utf-8").decode()
    doc = textacy.make_spacy_doc(overview, lang=en)

    keywords_algo_tri = textacy.extract.keyterms.sgrank(
        doc, topn=4, ngrams=(3))
    keywords_algo_tri = [i[0] for i in keywords_algo_tri]
    keywords_algo_bi = textacy.extract.keyterms.sgrank(doc, topn=5, ngrams=(2))
    keywords_algo_bi = [i[0] for i in keywords_algo_bi]
    keywords_algo_bi_filterd = []

    for bi in keywords_algo_bi:
        bool = True
        for tri in keywords_algo_tri:
            bool = bool and (not bi in tri)
        if bool:
            keywords_algo_bi_filterd.append(bi)

    artcle_keywords = list(set(getkeywords(url, graphids[article_type])))
    artcle_keywords = [i.lower() for i in artcle_keywords]
    final_cg_keywords = []
    overview_words = list(overview.split())
    for word in artcle_keywords:
        if " " in word and word in overview:
            final_cg_keywords.append(word)
        elif word in overview_words:
            final_cg_keywords.append(word)

    overview_words = [w for w in overview_words if not w.lower() in stop_words]
    overview = " ".join(overview_words)
    tokenization = nltk.word_tokenize(overview)
    overview = " ".join([wordnet_lemmatizer.lemmatize(w)
                        for w in tokenization])
    print(overview)
    doc = textacy.make_spacy_doc(overview, lang=en)
    keywords_algo_uni = textacy.extract.keyterms.sgrank(
        doc, topn=16, ngrams=(1))
    keywords_algo_uni = [i[0] for i in keywords_algo_uni]
    keywords_algo_uni_filterd = []

    for uni in keywords_algo_uni:
        bool = True
        for tri in keywords_algo_tri:
            bool = bool and (not uni in tri)
        for bi in keywords_algo_bi_filterd:
            bool = bool and (not uni in bi)
        if bool:
            keywords_algo_uni_filterd.append(uni)
    count = 0
    keywords_algo = keywords_algo_tri + \
        keywords_algo_bi_filterd+keywords_algo_uni_filterd
    inter_cg = set([])
    inter_algo = set([])
    for word in final_cg_keywords:
        for algo_word in keywords_algo:
            if word in algo_word or (word[-1] == 's' and word[:-1] in algo_word) or wordnet_lemmatizer.lemmatize(word) in algo_word or algo_word in wordnet_lemmatizer.lemmatize(word):
                inter_cg.add(word)
                inter_algo.add(algo_word)
                count += 1
                break
    if(len(overview_words) == 0 or len(final_cg_keywords) == 0 or len(keywords_algo) == 0):
        return
    compression = (len(keywords_algo)/len(overview_words))*100
    data['url'].append(url)
    data['overview'].append(overview)
    data['cg_keywords'].append(final_cg_keywords)
    data['algo_keywords'].append(keywords_algo)
    data['compressed percentage'].append(compression)
    data['cg-notAlgo'].append([i for i in final_cg_keywords if i not in inter_cg])
    algo_not_cg = [i for i in keywords_algo if i not in inter_algo]
    data['Algo-notCG'].append(algo_not_cg)
    percentage = (count/len(final_cg_keywords))*100
    data['percentage_match'].append(percentage)
    data['Algo_not_used'].append((len(algo_not_cg)/len(keywords_algo))*100)
    print(percentage)
    return percentage


for article_type in range(3):
    count = 0
    data = {'url': [], 'overview': [], 'cg_keywords': [], 'algo_keywords': [], 'cg-notAlgo': [],
            'Algo-notCG': [], 'percentage_match': [], 'compressed percentage': [], 'Algo_not_used': []}
    article_list = genrate_url(product_id[article_type], exp_ids[article_type])
    average_percent = 0
    shuffle(article_list)
    article_number = 0
    while count < 100 and article_number < 160:
        try:
            curr_article = article_list[article_number]
            average_percent += evaluate_article(curr_article,
                                                article_type, data)
            count += 1
        except:
            print("Overview not found")
        article_number += 1

    df = pd.DataFrame.from_dict(data)
    df.to_csv(f'analysis_{product_id[article_type]}-.csv')
    print((average_percent/count)*10000)
file1.close()
