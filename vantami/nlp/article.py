"""
A selection of classes and functions related to preparing a cardio toxicity review.
"""
import json
from lxml import html

from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# nltk.download("stopwords")
# nltk.download("punkt_tab")
# nltk.download("wordnet")


class Article:
    """
    Represent a scientific article and its metadata retrieved from PubMed Central.

    Parameters
    ----------
    verbose: int, default=0
        Verbosity level for error logging.
    """

    def __init__(self, verbose: int = 0):
        """
        Initialize an article instance with empty metadata fields.

        Parameters
        ----------
        verbose: int, default=0
            Verbosity level for error logging.
        """
        self.pmc_url = "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
        self.converter_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0"
        self.identifier = None
        self.metadata_type = None
        self.verbose = verbose

        self.pmcid = None
        self.pmid = None
        self.doi = None
        self.title = None
        self.journal = None
        self.journal_id = None
        self.journal_vol = None
        self.journal_issue = None
        self.pub_year = None
        self.pub_month = None
        self.pub_day = None
        self.issn = None
        self.publisher = None
        self.abstract = None
        self.tokens = None
        self.keywords = None

        self.conversion_json = None
        self.article_xml = None

    def __eq__(self, other):
        return self.pmcid == other.pmcid

    def __repr__(self):
        return f"{self.pmcid}_{self.doi}_{self.pmid}"

    def __str__(self):
        return f"{self.title}\n{self.doi}"

    def _log_error(self, error_type, error, context=""):
        if self.verbose > 0:
            print(f"{error_type} occurred in {context}:\n {error}")

    def from_pubmed_api(self, identifier, metadata_type: str = "pmc_fm"):
        """
        Populate article metadata using the PubMed Central API.

        Parameters
        ----------
        identifier: str
            Identifier to convert and use for metadata retrieval.
        metadata_type: str, default="pmc_fm"
            PubMed Central metadata format to request.
        """
        self.identifier = identifier
        self.metadata_type = metadata_type

        self.convert_ids()
        self.parse_conversion()

        self.collect_metadata()
        self.parse_metadata()

    def from_pubmed(self):
        raise NotImplementedError

    def from_ieee_xplore(self):
        raise NotImplementedError

    def from_embase(self):
        raise NotImplementedError

    def from_scopus(self):
        raise NotImplementedError

    def convert_ids(self):
        """
        Convert the article identifier to PMC, PubMed, and DOI identifiers.
        """
        url = f"{self.converter_url}/?ids={self.identifier}&tool=drid&email=mateusz.iwan%40hotmail.com&versions=no&format=json"
        try:
            response = urlopen(url, timeout=5).read().decode("utf-8")
            self.conversion_json = json.loads(response)
        except HTTPError as e:
            self._log_error("HTTPError", e, "convert_ids")
        except URLError as e:
            self._log_error("URLError", e, "convert_ids")
        except ConnectionResetError as e:
            self._log_error("ConnectionResetError", e, "convert_ids")
        except Exception as e:
            self._log_error("General Error", e, "convert_ids")

    def parse_conversion(self):
        """
        Extract converted identifiers from the conversion response.
        """
        if not self.conversion_json:
            self._log_error("Conversion Error", "No conversion data available")
            return
        try:
            records = self.conversion_json["records"][0]
            self.pmcid = records.get("pmcid")
            self.pmid = records.get("pmid")
            self.doi = records.get("doi")
        except KeyError as e:
            self._log_error("KeyError", e, "parse_conversion")
        except Exception as e:
            self._log_error("General Error", e, "parse_conversion")

    def collect_metadata(self):
        """
        Retrieve article metadata XML from PubMed Central.
        """
        if not self.pmcid:
            self._log_error("Metadata Collection Error", "PMCID is missing")
            return

        url = f'{self.pmc_url}?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:{str(self.pmcid).lstrip("PMC")}&metadataPrefix={self.metadata_type}'
        try:
            response = urlopen(url, data=None, timeout=5).read()
            self.article_xml = html.fromstring(response)
        except HTTPError as e:
            self._log_error("HTTPError", e, "collect_metadata")
        except URLError as e:
            self._log_error("URLError", e, "collect_metadata")
        except ConnectionResetError as e:
            self._log_error("ConnectionResetError", e, "collect_metadata")
        except Exception as e:
            self._log_error("General Error", e, "collect_metadata")

    def parse_metadata(self):
        """
        Extract article metadata fields from the retrieved XML.
        """
        if self.article_xml is None:
            self._log_error("Parsing Error", "No metadata XML to parse")
            return

        root = self.article_xml.xpath("./getrecord/record/metadata/article/front")
        if not root:
            self._log_error("Parsing Error", "Root element not found in article metadata")
            return

        root = root[0]

        self.title = self.fetch_element(root, "./article-meta/title-group/article-title", "title")
        self.journal = self.fetch_element(root, "./journal-meta/journal-title-group/journal-title", "journal")
        self.journal_id = self.fetch_element(root, "./journal-meta/journal-id", "journal_id")
        self.journal_vol = self.fetch_element(root, "./article-meta/volume", "journal_vol")
        self.journal_issue = self.fetch_element(root, "./article-meta/issue", "journal_issue")
        self.issn = self.fetch_element(root, "./journal-meta/issn", "issn")
        self.publisher = self.fetch_element(root, "./journal-meta/publisher", "publisher")
        self.pub_year = self.fetch_element(root, "./article-meta/pub-date/year", "year")
        self.pub_month = self.fetch_element(root, "./article-meta/pub-date/month", "month")
        self.pub_day = self.fetch_element(root, "./article-meta/pub-date/day", "day")
        self.abstract = self.fetch_element(root, "./article-meta/abstract/p", "abstract")
        if self.abstract == "":  # if abstract is not present within the usual path
            self.abstract = self.extract_text(root.xpath("./article-meta/abstract")[0])

    @staticmethod
    def fetch_element(element, xpath, name):
        """
        Extract text from the first XML element matching an XPath expression.

        Parameters
        ----------
        element: lxml.html.HtmlElement
            XML element to search.
        xpath: str
            XPath expression identifying the target element.
        name: str
            Field name used in error messages.

        Returns
        -------
        str
            Extracted text, or an empty string if extraction fails.
        """
        try:
            text = "".join(element.xpath(xpath)[0].itertext())
        except Exception as e:
            print(f"Error fetching {name}: {e}")
            text = ""
        return text

    def extract_text(self, element):
        """
        Recursively extract text content from an XML element.

        Parameters
        ----------
        element: lxml.html.HtmlElement
            XML element whose text content is extracted.

        Returns
        -------
        str
            Concatenated text from the element and its descendants.
        """
        text_content = []

        if element.text:
            text_content.append(element.text)

        for child in element:
            text_content.extend(self.extract_text(child))

        if element.tail:
            text_content.append(element.tail)

        return " ".join(text_content)

    def tokenize(self):
        """
        Tokenize, normalize, and extract keywords from the article abstract.
        """
        if not self.abstract:
            self._log_error("Tokenization Error", "No abstract to tokenize")
            return

        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()

        tokens = [word_tokenize(sent) for sent in sent_tokenize(self.abstract)]
        tokens = [token.lower() for sublist in tokens for token in sublist]
        tokens = [token for token in tokens if token.isalpha()]
        tokens = [token for token in tokens if token not in stop_words]
        self.tokens = [lemmatizer.lemmatize(token) for token in tokens]
        self.keywords = sorted(set(self.tokens))
