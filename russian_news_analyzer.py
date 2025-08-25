#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Russian News Titles Analyzer
Comprehensive ML pipeline for analyzing Russian news titles including:
- Lemmatization
- Sentiment Analysis (multiple dictionaries)
- Topic Modeling
- Named Entity Recognition
- Text Complexity Analysis
- Temporal Analysis
- Geographic Analysis
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Russian NLP
import pymorphy2
from razdel import tokenize
from natasha import (
    Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser, NewsNERTagger,
    NamesExtractor, DatesExtractor, MoneyExtractor, AddrExtractor
)

# Text processing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import gensim
from gensim import corpora, models
from gensim.models import LdaModel, CoherenceModel

# Sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textblob

# Topic modeling
from bertopic import BERTopic
import umap
import hdbscan

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium

# Additional analysis
import spacy
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

class RussianNewsAnalyzer:
    """
    Comprehensive analyzer for Russian news titles with multiple analysis capabilities
    """
    
    def __init__(self, language='ru'):
        """
        Initialize the analyzer with Russian language support
        
        Args:
            language (str): Language code (default: 'ru' for Russian)
        """
        self.language = language
        self.morph = pymorphy2.MorphAnalyzer()
        
        # Initialize Natasha components for NER
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.syntax_parser = NewsSyntaxParser(self.emb)
        self.ner_tagger = NewsNERTagger(self.emb)
        
        # Initialize extractors
        self.names_extractor = NamesExtractor(self.morph_vocab)
        self.dates_extractor = DatesExtractor(self.morph_vocab)
        self.money_extractor = MoneyExtractor(self.morph_vocab)
        self.addr_extractor = AddrExtractor(self.morph_vocab)
        
        # Initialize sentiment analyzer
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Russian stopwords
        self.russian_stopwords = set([
            'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она',
            'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее',
            'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда',
            'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до',
            'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей',
            'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем',
            'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет',
            'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь',
            'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем',
            'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть', 'после',
            'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много',
            'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда',
            'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю',
            'между', 'это', 'всё', 'всех', 'всего', 'всех', 'всего', 'всех', 'всего', 'всех', 'всего'
        ])
        
        # Political keywords for bias detection
        self.political_keywords = {
            'pro_government': ['путин', 'медведев', 'правительство', 'кремль', 'госдума', 'совет федерации'],
            'opposition': ['навальный', 'оппозиция', 'протест', 'митинг', 'демонстрация'],
            'international': ['сша', 'европа', 'нato', 'санкции', 'запад', 'восток'],
            'military': ['армия', 'военные', 'оборона', 'оружие', 'учения', 'война'],
            'economy': ['экономика', 'рубль', 'доллар', 'инфляция', 'бюджет', 'налоги']
        }
        
        # Emotion keywords
        self.emotion_keywords = {
            'anger': ['гнев', 'ярость', 'раздражение', 'возмущение', 'ненависть'],
            'fear': ['страх', 'ужас', 'паника', 'тревога', 'беспокойство'],
            'joy': ['радость', 'счастье', 'восторг', 'удовольствие', 'веселье'],
            'sadness': ['грусть', 'печаль', 'тоска', 'уныние', 'отчаяние'],
            'surprise': ['удивление', 'шок', 'изумление', 'поражение', 'неожиданность']
        }
        
        self.data = None
        self.processed_data = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load Russian news data from JSON file
        
        Args:
            file_path (str): Path to JSON file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f if line.strip()]
            
            df = pd.DataFrame(data)
            print(f"Loaded {len(df)} news articles")
            self.data = df
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize Russian text
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters but keep Russian letters
        text = re.sub(r'[^\w\sа-яёА-ЯЁ]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Convert to lowercase
        text = text.lower().strip()
        
        return text
    
    def lemmatize_text(self, text: str) -> str:
        """
        Lemmatize Russian text using pymorphy2
        
        Args:
            text (str): Clean text
            
        Returns:
            str: Lemmatized text
        """
        if not text:
            return ""
        
        # Tokenize using razdel (better for Russian)
        tokens = [token.text for token in tokenize(text)]
        
        # Lemmatize each token
        lemmatized = []
        for token in tokens:
            if token in self.russian_stopwords:
                continue
                
            # Get normal form using pymorphy2
            parsed = self.morph.parse(token)
            if parsed:
                lemma = parsed[0].normal_form
                lemmatized.append(lemma)
            else:
                lemmatized.append(token)
        
        return ' '.join(lemmatized)
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from Russian text
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, List[str]]: Extracted entities by type
        """
        entities = {
            'persons': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'money': []
        }
        
        try:
            # Process text with Natasha
            doc = self.segmenter(text)
            doc = self.morph_tagger(doc)
            doc = self.syntax_parser(doc)
            doc = self.ner_tagger(doc)
            
            # Extract entities
            for span in doc.spans:
                if span.type == 'PER':
                    entities['persons'].append(span.text)
                elif span.type == 'ORG':
                    entities['organizations'].append(span.text)
                elif span.type == 'LOC':
                    entities['locations'].append(span.text)
            
            # Extract dates and money
            doc = self.dates_extractor(doc)
            doc = self.money_extractor(doc)
            
            for span in doc.spans:
                if span.type == 'DATE':
                    entities['dates'].append(span.text)
                elif span.type == 'MONEY':
                    entities['money'].append(span.text)
                    
        except Exception as e:
            print(f"Error extracting entities: {e}")
            
        return entities
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using multiple approaches
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Sentiment scores
        """
        sentiment_scores = {}
        
        try:
            # VADER sentiment (works well with Russian)
            vader_scores = self.vader_analyzer.polarity_scores(text)
            sentiment_scores.update({
                'vader_compound': vader_scores['compound'],
                'vader_positive': vader_scores['pos'],
                'vader_negative': vader_scores['neg'],
                'vader_neutral': vader_scores['neu']
            })
            
            # TextBlob sentiment
            blob = textblob.TextBlob(text)
            sentiment_scores['textblob_polarity'] = blob.sentiment.polarity
            sentiment_scores['textblob_subjectivity'] = blob.sentiment.subjectivity
            
            # Custom Russian sentiment dictionary approach
            positive_words = ['хорошо', 'отлично', 'великолепно', 'успех', 'победа', 'достижение']
            negative_words = ['плохо', 'ужасно', 'провал', 'поражение', 'кризис', 'проблема']
            
            words = text.lower().split()
            pos_count = sum(1 for word in words if word in positive_words)
            neg_count = sum(1 for word in words if word in negative_words)
            
            sentiment_scores['custom_positive'] = pos_count
            sentiment_scores['custom_negative'] = neg_count
            sentiment_scores['custom_balance'] = pos_count - neg_count
            
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            
        return sentiment_scores
    
    def detect_emotions(self, text: str) -> Dict[str, float]:
        """
        Detect emotions in text using keyword matching
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Emotion scores
        """
        emotions = {}
        text_lower = text.lower()
        
        for emotion, keywords in self.emotion_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            emotions[f'emotion_{emotion}'] = count
            
        return emotions
    
    def detect_political_bias(self, text: str) -> Dict[str, float]:
        """
        Detect political bias using keyword analysis
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Political bias scores
        """
        bias_scores = {}
        text_lower = text.lower()
        
        for category, keywords in self.political_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            bias_scores[f'bias_{category}'] = count
            
        return bias_scores
    
    def analyze_text_complexity(self, text: str) -> Dict[str, float]:
        """
        Analyze text complexity metrics
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Complexity metrics
        """
        if not text:
            return {}
            
        words = text.split()
        sentences = text.split('.')
        
        metrics = {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            'unique_words': len(set(words)),
            'lexical_diversity': len(set(words)) / max(len(words), 1),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0
        }
        
        return metrics
    
    def extract_geographic_info(self, text: str) -> Dict[str, List[str]]:
        """
        Extract geographic information from text
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, List[str]]: Geographic information
        """
        # Russian cities and regions
        russian_cities = [
            'москва', 'санкт-петербург', 'новосибирск', 'екатеринбург', 'казань',
            'нижний новгород', 'челябинск', 'самара', 'ростов-на-дону', 'уфа',
            'волгоград', 'пермь', 'воронеж', 'омск', 'красноярск'
        ]
        
        # Russian regions
        russian_regions = [
            'московская область', 'ленинградская область', 'свердловская область',
            'краснодарский край', 'республика татарстан', 'республика башкортостан'
        ]
        
        text_lower = text.lower()
        cities = [city for city in russian_cities if city in text_lower]
        regions = [region for region in russian_regions if region in text_lower]
        
        return {
            'cities': cities,
            'regions': regions,
            'locations': self.extract_entities(text)['locations']
        }
    
    def process_titles(self) -> pd.DataFrame:
        """
        Process all news titles with comprehensive analysis
        
        Returns:
            pd.DataFrame: Processed data with all features
        """
        if self.data is None or self.data.empty:
            print("No data loaded. Please load data first.")
            return pd.DataFrame()
        
        print("Processing news titles...")
        processed_data = []
        
        for idx, row in self.data.iterrows():
            if idx % 100 == 0:
                print(f"Processing {idx}/{len(self.data)} titles...")
            
            title = row.get('title', '')
            if not title:
                continue
                
            # Clean and lemmatize
            clean_title = self.clean_text(title)
            lemmatized_title = self.lemmatize_text(clean_title)
            
            # Extract features
            entities = self.extract_entities(clean_title)
            sentiment = self.analyze_sentiment(clean_title)
            emotions = self.detect_emotions(clean_title)
            bias = self.detect_political_bias(clean_title)
            complexity = self.analyze_text_complexity(clean_title)
            geography = self.extract_geographic_info(clean_title)
            
            # Combine all features
            processed_row = {
                'original_title': title,
                'clean_title': clean_title,
                'lemmatized_title': lemmatized_title,
                'text': row.get('text', ''),
                **entities,
                **sentiment,
                **emotions,
                **bias,
                **complexity,
                **geography
            }
            
            processed_data.append(processed_row)
        
        self.processed_data = pd.DataFrame(processed_data)
        print(f"Processing complete. {len(self.processed_data)} titles processed.")
        return self.processed_data
    
    def perform_topic_modeling(self, method='lda', num_topics=10) -> Dict:
        """
        Perform topic modeling using various methods
        
        Args:
            method (str): Topic modeling method ('lda', 'nmf', 'bertopic')
            num_topics (int): Number of topics to extract
            
        Returns:
            Dict: Topic modeling results
        """
        if self.processed_data is None or self.processed_data.empty:
            print("No processed data available. Please process titles first.")
            return {}
        
        print(f"Performing {method.upper()} topic modeling...")
        
        if method == 'lda':
            return self._lda_topic_modeling(num_topics)
        elif method == 'nmf':
            return self._nmf_topic_modeling(num_topics)
        elif method == 'bertopic':
            return self._bertopic_modeling(num_topics)
        else:
            print(f"Unknown method: {method}")
            return {}
    
    def _lda_topic_modeling(self, num_topics: int) -> Dict:
        """LDA topic modeling implementation"""
        # Vectorize lemmatized titles
        vectorizer = CountVectorizer(
            max_features=1000,
            stop_words=list(self.russian_stopwords),
            min_df=2,
            max_df=0.95
        )
        
        X = vectorizer.fit_transform(self.processed_data['lemmatized_title'])
        feature_names = vectorizer.get_feature_names_out()
        
        # Fit LDA model
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            max_iter=50
        )
        
        lda.fit(X)
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-10:][::-1]]
            topics.append({
                'topic_id': topic_idx,
                'top_words': top_words,
                'weights': topic[topic.argsort()[-10:][::-1]].tolist()
            })
        
        # Assign topics to documents
        doc_topics = lda.transform(X)
        topic_assignments = doc_topics.argmax(axis=1)
        
        return {
            'method': 'LDA',
            'topics': topics,
            'topic_assignments': topic_assignments,
            'vectorizer': vectorizer,
            'model': lda
        }
    
    def _nmf_topic_modeling(self, num_topics: int) -> Dict:
        """NMF topic modeling implementation"""
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=list(self.russian_stopwords),
            min_df=2,
            max_df=0.95
        )
        
        X = vectorizer.fit_transform(self.processed_data['lemmatized_title'])
        feature_names = vectorizer.get_feature_names_out()
        
        # Fit NMF model
        nmf = NMF(n_components=num_topics, random_state=42, max_iter=200)
        nmf.fit(X)
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(nmf.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-10:][::-1]]
            topics.append({
                'topic_id': topic_idx,
                'top_words': top_words,
                'weights': topic[topic.argsort()[-10:][::-1]].tolist()
            })
        
        # Assign topics to documents
        doc_topics = nmf.transform(X)
        topic_assignments = doc_topics.argmax(axis=1)
        
        return {
            'method': 'NMF',
            'topics': topics,
            'topic_assignments': topic_assignments,
            'vectorizer': vectorizer,
            'model': nmf
        }
    
    def _bertopic_modeling(self, num_topics: int) -> Dict:
        """BERTopic modeling implementation"""
        try:
            # Prepare documents
            docs = self.processed_data['lemmatized_title'].tolist()
            
            # Initialize BERTopic
            topic_model = BERTopic(
                nr_topics=num_topics,
                random_state=42,
                verbose=True
            )
            
            # Fit and transform
            topics, probs = topic_model.fit_transform(docs)
            
            # Get topic info
            topic_info = topic_model.get_topic_info()
            
            return {
                'method': 'BERTopic',
                'topics': topic_info.to_dict('records'),
                'topic_assignments': topics,
                'model': topic_model
            }
            
        except Exception as e:
            print(f"BERTopic modeling failed: {e}")
            print("Falling back to LDA...")
            return self._lda_topic_modeling(num_topics)
    
    def create_visualizations(self, save_path: str = None):
        """
        Create comprehensive visualizations of the analysis
        
        Args:
            save_path (str): Path to save visualizations
        """
        if self.processed_data is None or self.processed_data.empty:
            print("No processed data available for visualization.")
            return
        
        print("Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Sentiment distribution
        plt.subplot(4, 2, 1)
        sentiment_cols = [col for col in self.processed_data.columns if 'vader_' in col and col != 'vader_compound']
        sentiment_data = self.processed_data[sentiment_cols].mean()
        sentiment_data.plot(kind='bar', ax=plt.gca())
        plt.title('Average Sentiment Scores (VADER)')
        plt.xticks(rotation=45)
        plt.ylabel('Score')
        
        # 2. Emotion distribution
        plt.subplot(4, 2, 2)
        emotion_cols = [col for col in self.processed_data.columns if 'emotion_' in col]
        emotion_data = self.processed_data[emotion_cols].sum()
        emotion_data.plot(kind='bar', ax=plt.gca())
        plt.title('Emotion Distribution')
        plt.xticks(rotation=45)
        plt.ylabel('Count')
        
        # 3. Political bias
        plt.subplot(4, 2, 3)
        bias_cols = [col for col in self.processed_data.columns if 'bias_' in col]
        bias_data = self.processed_data[bias_cols].sum()
        bias_data.plot(kind='bar', ax=plt.gca())
        plt.title('Political Bias Distribution')
        plt.xticks(rotation=45)
        plt.ylabel('Count')
        
        # 4. Text complexity
        plt.subplot(4, 2, 4)
        complexity_cols = ['word_count', 'avg_sentence_length', 'lexical_diversity']
        complexity_data = self.processed_data[complexity_cols].mean()
        complexity_data.plot(kind='bar', ax=plt.gca())
        plt.title('Text Complexity Metrics')
        plt.xticks(rotation=45)
        plt.ylabel('Value')
        
        # 5. Geographic distribution
        plt.subplot(4, 2, 5)
        all_cities = []
        for cities in self.processed_data['cities']:
            all_cities.extend(cities)
        city_counts = Counter(all_cities)
        top_cities = dict(city_counts.most_common(10))
        plt.bar(range(len(top_cities)), list(top_cities.values()))
        plt.title('Top 10 Cities Mentioned')
        plt.xticks(range(len(top_cities)), list(top_cities.keys()), rotation=45)
        plt.ylabel('Count')
        
        # 6. Sentiment over time (if date available)
        plt.subplot(4, 2, 6)
        if 'vader_compound' in self.processed_data.columns:
            plt.hist(self.processed_data['vader_compound'], bins=30, alpha=0.7)
            plt.title('Sentiment Distribution (Compound Score)')
            plt.xlabel('Sentiment Score')
            plt.ylabel('Frequency')
        
        # 7. Word length distribution
        plt.subplot(4, 2, 7)
        if 'avg_word_length' in self.processed_data.columns:
            plt.hist(self.processed_data['avg_word_length'], bins=20, alpha=0.7)
            plt.title('Average Word Length Distribution')
            plt.xlabel('Average Word Length')
            plt.ylabel('Frequency')
        
        # 8. Topic distribution (if available)
        plt.subplot(4, 2, 8)
        plt.text(0.5, 0.5, 'Topic distribution\n(available after topic modeling)', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Topic Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualizations saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, output_path: str = None) -> str:
        """
        Generate comprehensive analysis report
        
        Args:
            output_path (str): Path to save report
            
        Returns:
            str: Report content
        """
        if self.processed_data is None or self.processed_data.empty:
            return "No data available for report generation."
        
        report = []
        report.append("=" * 80)
        report.append("RUSSIAN NEWS TITLES ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Dataset overview
        report.append("DATASET OVERVIEW")
        report.append("-" * 40)
        report.append(f"Total articles analyzed: {len(self.processed_data)}")
        report.append(f"Average title length: {self.processed_data['word_count'].mean():.1f} words")
        report.append(f"Average sentence length: {self.processed_data['avg_sentence_length'].mean():.1f} words")
        report.append("")
        
        # Sentiment analysis
        report.append("SENTIMENT ANALYSIS")
        report.append("-" * 40)
        if 'vader_compound' in self.processed_data.columns:
            compound_scores = self.processed_data['vader_compound']
            positive = (compound_scores > 0.05).sum()
            negative = (compound_scores < -0.05).sum()
            neutral = ((compound_scores >= -0.05) & (compound_scores <= 0.05)).sum()
            
            report.append(f"Positive titles: {positive} ({positive/len(compound_scores)*100:.1f}%)")
            report.append(f"Negative titles: {negative} ({negative/len(compound_scores)*100:.1f}%)")
            report.append(f"Neutral titles: {neutral} ({neutral/len(compound_scores)*100:.1f}%)")
            report.append(f"Average sentiment score: {compound_scores.mean():.3f}")
        report.append("")
        
        # Political bias
        report.append("POLITICAL BIAS ANALYSIS")
        report.append("-" * 40)
        bias_cols = [col for col in self.processed_data.columns if 'bias_' in col]
        for col in bias_cols:
            count = (self.processed_data[col] > 0).sum()
            report.append(f"{col.replace('bias_', '').title()}: {count} titles")
        report.append("")
        
        # Geographic analysis
        report.append("GEOGRAPHIC ANALYSIS")
        report.append("-" * 40)
        all_cities = []
        for cities in self.processed_data['cities']:
            all_cities.extend(cities)
        city_counts = Counter(all_cities)
        top_cities = city_counts.most_common(5)
        report.append("Top 5 cities mentioned:")
        for city, count in top_cities:
            report.append(f"  {city}: {count} mentions")
        report.append("")
        
        # Text complexity
        report.append("TEXT COMPLEXITY ANALYSIS")
        report.append("-" * 40)
        report.append(f"Average lexical diversity: {self.processed_data['lexical_diversity'].mean():.3f}")
        report.append(f"Average word length: {self.processed_data['avg_word_length'].mean():.1f} characters")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        report.append("1. Consider temporal analysis to track sentiment trends over time")
        report.append("2. Implement entity linking for better organization and person tracking")
        report.append("3. Add multilingual sentiment analysis for international news")
        report.append("4. Consider adding fact-checking indicators")
        report.append("5. Implement source credibility scoring")
        report.append("")
        
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Report saved to {output_path}")
        
        return report_text

def main():
    """Main function to demonstrate the analyzer"""
    print("Russian News Titles Analyzer")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = RussianNewsAnalyzer()
    
    # Load data
    print("Loading data...")
    data = analyzer.load_data('ria_20.json')
    
    if data.empty:
        print("Failed to load data. Exiting.")
        return
    
    # Process titles
    processed_data = analyzer.process_titles()
    
    # Perform topic modeling
    print("\nPerforming topic modeling...")
    lda_results = analyzer.perform_topic_modeling('lda', num_topics=5)
    
    if lda_results:
        print("\nLDA Topics:")
        for topic in lda_results['topics']:
            print(f"Topic {topic['topic_id']}: {', '.join(topic['top_words'][:5])}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    analyzer.create_visualizations()
    
    # Generate report
    print("\nGenerating report...")
    report = analyzer.generate_report('analysis_report.txt')
    print("\nReport preview:")
    print(report[:500] + "...")

if __name__ == "__main__":
    main()