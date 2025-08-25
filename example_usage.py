#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage of Russian News Analyzer
Simple demonstration of key features
"""

from russian_news_analyzer import RussianNewsAnalyzer
import pandas as pd

def simple_example():
    """Simple example of using the analyzer"""
    print("=== Russian News Analyzer - Simple Example ===\n")
    
    # Initialize analyzer
    analyzer = RussianNewsAnalyzer()
    
    # Load sample data
    print("1. Loading data...")
    data = analyzer.load_data('ria_20.json')
    
    if data.empty:
        print("❌ Failed to load data")
        return
    
    print(f"✅ Loaded {len(data)} articles\n")
    
    # Show sample titles
    print("2. Sample titles:")
    for i, title in enumerate(data['title'].head(3)):
        print(f"   {i+1}. {title}")
    print()
    
    # Process titles (this will take a moment)
    print("3. Processing titles (lemmatization, sentiment, etc.)...")
    processed = analyzer.process_titles()
    
    if processed.empty:
        print("❌ Failed to process titles")
        return
    
    print(f"✅ Processed {len(processed)} titles\n")
    
    # Show some results
    print("4. Analysis Results:")
    
    # Sentiment
    if 'vader_compound' in processed.columns:
        avg_sentiment = processed['vader_compound'].mean()
        print(f"   Average sentiment: {avg_sentiment:.3f}")
        
        positive = (processed['vader_compound'] > 0.05).sum()
        negative = (processed['vader_compound'] < -0.05).sum()
        print(f"   Positive titles: {positive}")
        print(f"   Negative titles: {negative}")
    
    # Text complexity
    if 'word_count' in processed.columns:
        avg_words = processed['word_count'].mean()
        print(f"   Average words per title: {avg_words:.1f}")
    
    # Political bias
    bias_cols = [col for col in processed.columns if 'bias_' in col]
    if bias_cols:
        print("   Political bias detected in categories:")
        for col in bias_cols:
            count = (processed[col] > 0).sum()
            if count > 0:
                category = col.replace('bias_', '').replace('_', ' ').title()
                print(f"     {category}: {count} titles")
    
    print()
    
    # Topic modeling
    print("5. Topic Modeling (LDA)...")
    topics = analyzer.perform_topic_modeling('lda', num_topics=3)
    
    if topics and 'topics' in topics:
        print("   Topics found:")
        for topic in topics['topics']:
            words = ', '.join(topic['top_words'][:5])
            print(f"     Topic {topic['topic_id']}: {words}")
    
    print("\n✅ Analysis complete!")
    
    # Save results
    processed.to_csv('processed_news.csv', index=False, encoding='utf-8')
    print("💾 Results saved to 'processed_news.csv'")

def advanced_example():
    """Advanced example with custom analysis"""
    print("\n=== Advanced Analysis Example ===\n")
    
    analyzer = RussianNewsAnalyzer()
    data = analyzer.load_data('ria_20.json')
    
    if data.empty:
        return
    
    # Process titles
    processed = analyzer.process_titles()
    
    # Custom analysis
    print("Custom Analysis Results:")
    
    # Find most emotional titles
    if 'emotion_fear' in processed.columns:
        fear_titles = processed[processed['emotion_fear'] > 0]
        if not fear_titles.empty:
            print(f"\nTitles with fear emotion ({len(fear_titles)} found):")
            for _, row in fear_titles.head(3).iterrows():
                print(f"  - {row['original_title']}")
    
    # Geographic analysis
    all_cities = []
    for cities in processed['cities']:
        all_cities.extend(cities)
    
    if all_cities:
        from collections import Counter
        city_counts = Counter(all_cities)
        print(f"\nTop cities mentioned:")
        for city, count in city_counts.most_common(3):
            print(f"  {city}: {count} mentions")
    
    # Generate comprehensive report
    print("\nGenerating comprehensive report...")
    report = analyzer.generate_report('detailed_report.txt')
    print("📊 Report generated and saved to 'detailed_report.txt'")

if __name__ == "__main__":
    try:
        simple_example()
        advanced_example()
        
        print("\n🎉 All examples completed successfully!")
        print("\nFiles created:")
        print("  - processed_news.csv (processed data)")
        print("  - detailed_report.txt (analysis report)")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")