# The Sims 4 Community Analysis: Sentiment, Topics & Player Experience Insights

This project combines **data science, NLP, and product thinking** to analyze how *The Sims 4* player community has evolved over the past decade. By processing **591K+ Reddit posts and comments**, the analysis reveals long-term sentiment trends and key topics that drive player satisfaction, frustration, and engagement.

### Project Overview
Player communities generate enormous volumes of unstructured feedback. This project demonstrates how scalable NLP pipelines can turn that data into **actionable insights** for both research and real-world product improvement.

It integrates sarcasm-aware sentiment analysis, BERTopic modeling, and SQL-based data engineering to extract interpretable patterns from noisy social media text — all while maintaining research transparency and reproducibility.

### Key Highlights
- **Dataset:** 591,584 Reddit posts and comments (2014–2024) from Sims-related subreddits  
- **Tech Stack:** Python, pandas, spaCy, NLTK, VADER, BERTopic, SQL  
- **Approach:** Text cleaning, sarcasm detection, sentiment classification, and topic modeling  
- **Outputs:**  
  - 42 discussion clusters condensed into 5 high-impact opportunity areas  
  - Year-over-year sentiment tracking for community health analysis  
  - Reproducible notebook pipeline for future topic trend updates  

### Key Findings
- *Gameplay depth* and *AI behavior* consistently rank among the most discussed (and negatively scored) topics.  
- Sentiment toward *representation and diversity* has trended positive since 2020, though engagement has plateaued.  
- Incorporating sarcasm-sensitive models improved classification accuracy by **8%** compared to standard sentiment tools.  

### You can try this project out yourself at [THIS](https://drive.google.com/drive/folders/15JUEyK4gF8M8-KGQmXtMqhfX3Gs_1vyO?usp=sharing) link.

### Future Directions
- Expand to full Reddit comment trees (≈600K+ additional records)  
- Integrate time-aware embeddings for longitudinal topic sentiment tracking  
- Compare BERTopic results to transformer-based topic models (e.g., BERTopic + BERTopic-CTFIDF hybrid)  

---

*Author: [Bader Rezek](https://github.com/BaderRezek)*  
*University of Illinois Chicago | Data Science
