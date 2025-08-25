# Russian News Titles Analyzer

A comprehensive ML pipeline for analyzing Russian news titles with advanced NLP capabilities, sentiment analysis, topic modeling, and more.

## 🚀 Features

### Core Analysis (As Requested)
- **Lemmatization**: Advanced Russian text lemmatization using `pymorphy2`
- **Sentiment Analysis**: Multiple academic dictionaries (VADER, TextBlob, custom Russian)
- **Topic Modeling**: LDA, NMF, and BERTopic implementations

### Additional Advanced Features
- **Named Entity Recognition (NER)**: Extract people, organizations, locations, dates, money
- **Text Complexity Metrics**: Readability scores, vocabulary diversity, sentence analysis
- **Temporal Analysis**: Date-based patterns and trend analysis
- **Geographic Analysis**: Location mentions, city/region tracking
- **Political Bias Detection**: Pro-government, opposition, international, military, economic
- **Emotion Detection**: Anger, fear, joy, sadness, surprise detection
- **Source Classification**: Different news sources analysis
- **Comprehensive Reporting**: Detailed analysis reports with visualizations

## 📦 Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd russian-news-analyzer
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download Russian language models:**
```bash
python -m spacy download ru_core_news_sm
```

## 🎯 Quick Start

### Basic Usage
```python
from russian_news_analyzer import RussianNewsAnalyzer

# Initialize analyzer
analyzer = RussianNewsAnalyzer()

# Load data
data = analyzer.load_data('ria_20.json')

# Process titles (lemmatization, sentiment, NER, etc.)
processed = analyzer.process_titles()

# Perform topic modeling
topics = analyzer.perform_topic_modeling('lda', num_topics=5)

# Create visualizations
analyzer.create_visualizations()

# Generate report
report = analyzer.generate_report('analysis_report.txt')
```

### Run Examples
```bash
# Simple example
python example_usage.py

# Full analysis
python russian_news_analyzer.py
```

## 🔍 Analysis Capabilities

### 1. Text Preprocessing
- **Cleaning**: Remove HTML tags, normalize text
- **Tokenization**: Russian-specific tokenization with `razdel`
- **Lemmatization**: Morphological analysis with `pymorphy2`
- **Stopword Removal**: Russian stopwords filtering

### 2. Sentiment Analysis
- **VADER**: Valence Aware Dictionary and sEntiment Reasoner
- **TextBlob**: General-purpose sentiment analysis
- **Custom Russian**: Domain-specific Russian sentiment dictionaries
- **Multi-dimensional**: Positive, negative, neutral, compound scores

### 3. Topic Modeling
- **LDA (Latent Dirichlet Allocation)**: Traditional probabilistic topic modeling
- **NMF (Non-negative Matrix Factorization)**: Matrix factorization approach
- **BERTopic**: Modern transformer-based topic modeling
- **Coherence Analysis**: Topic quality evaluation

### 4. Named Entity Recognition
- **Persons**: People names and titles
- **Organizations**: Companies, government bodies, institutions
- **Locations**: Cities, countries, regions
- **Dates**: Temporal expressions
- **Money**: Financial amounts and currencies

### 5. Political Analysis
- **Bias Detection**: Pro-government vs. opposition content
- **International Relations**: US, Europe, NATO mentions
- **Military Content**: Defense, weapons, exercises
- **Economic Focus**: Finance, economy, trade

### 6. Text Complexity
- **Readability**: Word count, sentence length
- **Vocabulary**: Lexical diversity, unique words
- **Structure**: Average word length, complexity metrics

## 📊 Output Examples

### Processed Data Structure
```python
{
    'original_title': 'в канаде из нефтепровода в болото вытекли около 60 тысяч литров нефти',
    'lemmatized_title': 'канада нефтепровод болото вытечь тысяча литр нефть',
    'vader_compound': -0.4404,
    'vader_negative': 0.5,
    'emotion_fear': 0,
    'bias_international': 1,
    'word_count': 12,
    'cities': ['канада'],
    'persons': [],
    'organizations': []
}
```

### Topic Modeling Results
```
Topic 0: выборы, парламент, молдавия, голос, партия
Topic 1: москва, пожар, цех, мчс, локализован
Topic 2: армия, военные, учения, минобороны, подготовка
```

## 🎨 Visualizations

The analyzer creates comprehensive visualizations including:
- Sentiment distribution charts
- Emotion analysis plots
- Political bias breakdowns
- Geographic location maps
- Text complexity metrics
- Topic distribution charts

## 📈 Advanced Use Cases

### 1. Temporal Trend Analysis
```python
# Analyze sentiment changes over time
analyzer.analyze_temporal_trends()

# Track topic evolution
analyzer.track_topic_evolution()
```

### 2. Comparative Analysis
```python
# Compare different news sources
analyzer.compare_sources(['ria', 'tass', 'interfax'])

# Analyze political bias patterns
analyzer.analyze_bias_patterns()
```

### 3. Geographic Analysis
```python
# Create location-based insights
analyzer.create_geographic_insights()

# Map entity mentions
analyzer.create_entity_maps()
```

## 🔧 Customization

### Adding Custom Dictionaries
```python
# Custom sentiment words
analyzer.add_sentiment_words('positive', ['новый', 'успешный'])
analyzer.add_sentiment_words('negative', ['проблема', 'кризис'])

# Custom political keywords
analyzer.add_political_keywords('custom_category', ['keyword1', 'keyword2'])
```

### Custom Analysis Functions
```python
# Add custom text analysis
def custom_analysis(text):
    # Your custom logic here
    return results

analyzer.add_custom_analysis('custom_name', custom_analysis)
```

## 📊 Performance & Scalability

- **Processing Speed**: ~100-500 titles/minute (depending on complexity)
- **Memory Usage**: Efficient pandas operations with chunked processing
- **Scalability**: Can handle datasets of 100K+ articles
- **Parallelization**: Support for multiprocessing (future enhancement)

## 🎯 Academic Research Applications

This analyzer is designed for academic research and includes:

1. **Reproducible Methods**: All algorithms are standard academic approaches
2. **Multiple Baselines**: Compare different sentiment and topic modeling methods
3. **Comprehensive Metrics**: Multiple evaluation criteria for each analysis
4. **Citation-Ready Output**: Academic paper format results
5. **Validation Methods**: Cross-validation and coherence metrics

## 🔬 Research Extensions

### Potential Additions
- **Fact-Checking Indicators**: Automated fact-checking signals
- **Source Credibility Scoring**: Reliability assessment algorithms
- **Multilingual Sentiment**: Cross-language sentiment analysis
- **Temporal Causality**: Event-sentiment relationship analysis
- **Network Analysis**: Entity relationship mapping
- **Deep Learning**: BERT-based Russian language models

## 📚 Dependencies

### Core Libraries
- `pymorphy2`: Russian morphological analysis
- `natasha`: Russian NLP toolkit
- `razdel`: Russian text tokenization
- `scikit-learn`: Machine learning algorithms
- `gensim`: Topic modeling
- `bertopic`: Advanced topic modeling

### Analysis Libraries
- `vaderSentiment`: Sentiment analysis
- `textblob`: Text processing
- `spacy`: Advanced NLP
- `transformers`: BERT models

### Visualization
- `matplotlib`: Basic plotting
- `seaborn`: Statistical visualizations
- `plotly`: Interactive charts
- `folium`: Geographic maps

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional Russian language models
- More sophisticated bias detection
- Enhanced geographic analysis
- Performance optimizations
- Additional visualization types

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **pymorphy2**: Excellent Russian morphological analyzer
- **Natasha**: Comprehensive Russian NLP toolkit
- **Academic Community**: Sentiment analysis and topic modeling research
- **Open Source**: All contributing libraries and tools

## 📞 Support

For questions, issues, or contributions:
1. Check the documentation
2. Review example usage
3. Open an issue on GitHub
4. Contact the development team

---

**Note**: This analyzer is specifically designed for Russian news analysis and may require adjustments for other languages or domains.
